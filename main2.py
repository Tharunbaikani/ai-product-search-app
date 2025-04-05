import streamlit as st
import numpy as np
import pandas as pd
import copy
import json
import openai
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from transformers import pipeline
import time
import concurrent.futures
from typing import List, Dict, Any

# Configure OpenAI API key
openai.api_key = st.secrets["openai_api_key"]
# Configure Pinecone
pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
INDEX_NAME = "product-search"

# --- Download required NLTK data ---
if 'nltk_downloaded' not in st.session_state:
    for resource in ['wordnet', 'punkt']:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)
    st.session_state.nltk_downloaded = True

# --- Helper Functions ---

def get_openai_embedding(text):
    """Get embedding from OpenAI API with retry logic and rate limiting."""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            st.error(f"Error getting OpenAI embedding after {max_retries} attempts: {str(e)}")
            return None

def clean_description(description):
    """Clean description by removing HTML tags and formatting."""
    if not isinstance(description, str):
        return ""
    
    # Remove HTML tags
    description = description.replace('<ul>', '').replace('</ul>', '')
    description = description.replace('<li>', '').replace('</li>', '')
    description = description.replace('<p>', '').replace('</p>', '')
    description = description.replace('<b>', '').replace('</b>', '')
    
    # Remove JSON-like tags and formatting
    description = description.replace('"', '').replace('{', '').replace('}', '')
    description = description.replace('tags:', '').replace('category:', '')
    description = description.replace('name:', '').replace('description:', '')
    description = description.replace('slug:', '').replace('sku:', '')
    
    # Remove extra whitespace and newlines
    description = ' '.join(description.split())
    
    # Extract structured data
    structured_data = {}
    lines = description.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                structured_data[key] = value
        else:
            cleaned_lines.append(line)
    
    # Create a clean description
    clean_desc = ' '.join(cleaned_lines)
    
    # Add structured data in a clean format
    if structured_data:
        clean_desc += "\n\nProduct Details:"
        for key, value in structured_data.items():
            clean_desc += f"\n{key}: {value}"
    
    return clean_desc.strip()

def create_rich_context(product):
    """Create rich context for embedding with optimized information."""
    # Clean up description
    description = clean_description(product.get('description', ''))
    
    # Extract key features
    key_features = extract_key_features(description)
    
    # Get category and tags safely
    category = product.get('category', 'Uncategorized')
    tags = product.get('tags', [])
    
    # Create a rich context that includes all relevant information
    context = f"""
    Product Name: {product.get('name', '')}
    Description: {description}
    Category: {category}
    Tags: {', '.join(tags)}
    Key Features: {key_features}
    """
    return context

def extract_key_features(description):
    """Extract key features from description for better context."""
    # Split into lines and clean
    lines = description.split('\n')
    features = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for key-value pairs
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                features.append(f"{key}: {value}")
        # Look for bullet points or list items
        elif line.startswith(('-', '*', '•')):
            features.append(line.lstrip('-*• ').strip())
        # Look for sentences with important keywords
        elif any(keyword in line.lower() for keyword in ['feature', 'benefit', 'advantage', 'include', 'offer']):
            features.append(line)
    
    return ' | '.join(features) if features else description[:200]  # Fallback to first 200 chars if no features found

def init_pinecone():
    """Initialize Pinecone index."""
    try:
        # Create index if it doesn't exist
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,  # OpenAI ada-002 embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Get the index
        index = pc.Index(INDEX_NAME)
        
        # Verify index is ready
        if not index:
            st.error("Failed to get Pinecone index")
            return None
            
        # Get index stats to verify
        stats = index.describe_index_stats()
        st.info(f"Pinecone index initialized. Current vector count: {stats.total_vector_count}")
        
        return index
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return None

def update_product_embedding(product):
    """Update the embedding for a given product with optimized context."""
    try:
        # Create rich context with updated description and name
        rich_context = f"""
        Product Name: {product.get('name', '')}
        Description: {clean_description(product.get('description', ''))}
        """
        
        # Get new embedding
        emb = get_openai_embedding(rich_context)
        if not emb:
            st.error("Failed to get embedding from OpenAI")
            return False
            
        # Update Pinecone
        index = init_pinecone()
        if not index:
            st.error("Failed to initialize Pinecone index")
            return False
            
        # Prepare metadata with updated description
        metadata = {
            "name": product.get('name', ''),
            "description": product.get('description', ''),
            "category": product.get('category', 'Uncategorized'),
            "tags": product.get('tags', []),
            "slug": product.get('slug', ''),
            "sku": product.get('sku', f"SKU_{product.get('id', '')}"),
            "last_updated": time.time()  # Add timestamp for tracking updates
        }
        
        # Upsert to Pinecone with updated data
        try:
            index.upsert(
                vectors=[(str(product.get('id', '')), emb, metadata)],
                namespace="products"
            )
            st.success(f"Successfully updated embedding for product {product.get('id', '')}: {product.get('name', '')}")
            return True
                
        except Exception as e:
            st.error(f"Error upserting to Pinecone: {str(e)}")
            return False
            
    except Exception as e:
        st.error(f"Error in update_product_embedding: {str(e)}")
        return False

def load_services_data():
    """Load services data from JSON file."""
    try:
        # Try UTF-8 encoding first
        with open('services_sample_data.json', 'r', encoding='utf-8') as f:
            services_data = json.load(f)
        return services_data
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try with utf-8-sig (for files with BOM)
            with open('services_sample_data.json', 'r', encoding='utf-8-sig') as f:
                services_data = json.load(f)
            return services_data
        except Exception as e:
            st.error(f"Error loading services data with UTF-8-SIG: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error loading services data: {str(e)}")
        return None

def save_services_data(products):
    """Save updated products back to the JSON file."""
    try:
        # Convert products dictionary to list format
        services_data = []
        for product_id in sorted(products.keys()):
            product = products[product_id]
            # Ensure all required fields are present with default values
            services_data.append({
                "id": product.get('id', product_id),
                "name": product.get('name', ''),
                "slug": product.get('slug', f"product-{product_id}"),
                "description": product.get('description', ''),
                "category": product.get('category', 'Uncategorized'),
                "tags": product.get('tags', []),
                "sku": product.get('sku', f"SKU_{product_id}")
            })
        
        # Save to JSON file
        with open('services_sample_data.json', 'w', encoding='utf-8') as f:
            json.dump(services_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving services data: {str(e)}")
        return False

def process_batch(batch_products: List[Dict[str, Any]], index) -> List[Dict[str, Any]]:
    """Process a batch of products in parallel."""
    results = []
    failed_products = []
    
    for product in batch_products:
        try:
            # Extract primary category and tags from the weighted lists
            primary_category = product.get('category', 'Uncategorized')
            tags = product.get('tags', [])
            
            # Create rich context with updated description
            rich_context = create_rich_context({
                'name': product.get('name', ''),
                'description': product.get('description', ''),
                'category': primary_category,
                'tags': tags
            })
            
            emb = get_openai_embedding(rich_context)
            
            if emb:
                # Prepare metadata
                metadata = {
                    "name": product.get('name', ''),
                    "description": product.get('description', ''),
                    "category": primary_category,
                    "tags": tags,
                    "slug": product.get('slug', ''),
                    "sku": product.get('sku', f"SKU_{product.get('id', '')}"),
                    "all_categories": [cat['category'] for cat in product.get('ai_prod_categories', [])] if 'ai_prod_categories' in product else [],
                    "all_tags": [tag['tag'] for tag in product.get('ai_prod_tags', [])] if 'ai_prod_tags' in product else []
                }
                
                # Upsert to Pinecone
                index.upsert(
                    vectors=[(str(product.get('id', '')), emb, metadata)],
                    namespace="products"
                )
                results.append(product)
                st.write(f"Successfully processed product {product.get('id', '')}: {product.get('name', '')}")
            else:
                st.error(f"Failed to get embedding for product {product.get('id', '')}: {product.get('name', '')}")
                failed_products.append(product)
                
        except Exception as e:
            st.error(f"Error processing product {product.get('id', '')}: {str(e)}")
            st.error(f"Product details: {product}")
            failed_products.append(product)
    
    if failed_products:
        st.error(f"Failed to process {len(failed_products)} products in this batch:")
        for product in failed_products:
            st.error(f"- ID: {product.get('id', '')}, Name: {product.get('name', '')}")
    
    return results

def init_data():
    """Initialize data from services JSON and Pinecone with optimized batch processing."""
    # Show progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load services data
    status_text.text("Loading products...")
    services_data = load_services_data()
    if not services_data:
        st.error("Failed to load services data. Please check the file.")
        return False
    
    # Process ALL products
    total_products = len(services_data)
    st.info(f"Found {total_products} products in services data")
    
    progress_bar.progress(0.2)  # 20% complete
    status_text.text("Initializing products...")
    
    # Initialize products dictionary with ALL products
    st.session_state.products = {}
    for idx, product in enumerate(services_data):
        # Add ID if not present
        if 'id' not in product:
            product['id'] = idx
        st.session_state.products[product['id']] = product
    
    # Debug logging
    st.write(f"Session state products count: {len(st.session_state.products)}")
    
    progress_bar.progress(0.4)  # 40% complete
    status_text.text("Checking Pinecone index...")
    
    # Initialize Pinecone index once
    index = init_pinecone()
    if not index:
        st.error("Failed to initialize Pinecone index")
        return False
    
    # Get current Pinecone stats
    try:
        current_stats = index.describe_index_stats()
        current_count = current_stats.total_vector_count
        st.info(f"Current Pinecone index has {current_count} vectors")
        
        # If we already have 1345 products, skip the loading process
        if current_count >= 1345:
            st.success(f"Pinecone index already has {current_count} products. Skipping loading process.")
            progress_bar.progress(1.0)
            return True
            
        # Only process if we need more vectors
        if current_count < total_products:
            # Process embeddings in smaller batches with parallel processing
            BATCH_SIZE = 50  # Further reduced batch size for better reliability
            processed_count = current_count
            all_products = list(st.session_state.products.values())
            failed_products = []
            
            # Create batches of remaining products
            batches = []
            for i in range(current_count, total_products, BATCH_SIZE):
                batch = all_products[i:i + BATCH_SIZE]
                batches.append(batch)
            
            # Debug logging
            st.write(f"Number of batches to process: {len(batches)}")
            
            # Process batches sequentially for better reliability
            for batch in batches:
                try:
                    results = process_batch(batch, index)
                    processed_count += len(results)
                    # Calculate progress between 0.4 and 0.9 (40% to 90%)
                    progress = 0.4 + (processed_count / total_products * 0.5)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing embeddings: {processed_count}/{total_products}")
                    
                    # Verify batch was processed
                    if len(results) != len(batch):
                        st.warning(f"Batch processing incomplete: {len(results)}/{len(batch)} products processed")
                        failed_products.extend([p for p in batch if p not in results])
                    
                    # Add a small delay between batches to avoid rate limits
                    time.sleep(1)
                    
                except Exception as e:
                    st.error(f"Error processing batch: {str(e)}")
                    continue
            
            # Report failed products
            if failed_products:
                st.error(f"Total failed products: {len(failed_products)}")
                st.error("Failed product details:")
                for product in failed_products:
                    st.error(f"- ID: {product.get('id', '')}, Name: {product.get('name', '')}")
        else:
            st.success(f"Pinecone index already has all {current_count} products")
            progress_bar.progress(1.0)
            return True
    
    except Exception as e:
        st.error(f"Error checking Pinecone stats: {str(e)}")
        return False
    
    # Verify final Pinecone index
    try:
        final_stats = index.describe_index_stats()
        final_count = final_stats.total_vector_count
        
        if final_count != total_products:
            st.warning(f"Pinecone index count mismatch. Expected {total_products}, got {final_count}")
            st.warning("Some products may not have been processed correctly, but continuing with available products")
            # Don't return False here, just continue with what we have
            
        st.success(f"Successfully processed {final_count} products")
        
        # Additional verification - only check a sample of products
        st.write("Verifying product storage...")
        sample_size = min(50, final_count)  # Check up to 50 products
        sample_products = all_products[:sample_size]
        
        for product in sample_products:
            try:
                # Check if product exists in Pinecone
                results = index.query(
                    vector=get_openai_embedding(create_rich_context(product)),
                    top_k=1,
                    include_metadata=True,
                    namespace="products"
                )
                if not results.matches or str(product['id']) != results.matches[0].id:
                    st.warning(f"Product {product['id']} not found in Pinecone")
                    # Don't return False here, just continue
            except Exception as e:
                st.warning(f"Error verifying product {product['id']}: {str(e)}")
                # Don't return False here, just continue
        
        st.success("Product verification complete")
        
    except Exception as e:
        st.error(f"Error verifying Pinecone index: {str(e)}")
        # Don't return False here, just continue
    
    progress_bar.progress(1.0)  # 100% complete
    status_text.text(f"Initialization complete! Using {final_count} products")
    return True

def process_query(query):
    """Process the query with spell correction and synonym handling."""
    if 'spell' not in st.session_state:
        st.session_state.spell = SpellChecker()
        # Add product names and categories to the spell checker's dictionary
        for product in st.session_state.products.values():
            st.session_state.spell.word_frequency.load_text(product.get('name', ''))
            st.session_state.spell.word_frequency.load_text(product.get('category', 'Uncategorized'))
            for tag in product.get('tags', []):
                st.session_state.spell.word_frequency.load_text(tag)
    
    words = query.split()
    corrected_words = []
    for word in words:
        correction = st.session_state.spell.correction(word)
        corrected_words.append(correction)
    
    corrected_query = ' '.join(corrected_words)
    
    if corrected_query != query:
        st.info(f"Original query: '{query}'")
        st.info(f"Corrected query: '{corrected_query}'")
    
    return corrected_query

def get_relevant_products(query, top_k=20):
    """Get relevant products based on query using Pinecone with semantic search."""
    try:
        # First, use OpenAI to understand the query intent and extract key information
        query_analysis_prompt = f"""
        Analyze this product search query and extract key information:
        Query: "{query}"
        
        Extract:
        1. Main product type (e.g., toilet paper, tissue paper)
        2. Usage context (e.g., office, mall, home)
        3. Specific requirements or preferences
        4. Related product types
        
        Format as JSON:
        {{
            "product_type": "main product type",
            "usage_context": ["context1", "context2"],
            "requirements": ["req1", "req2"],
            "related_products": ["related1", "related2"]
        }}
        """
        
        try:
            query_analysis = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a product search assistant that analyzes queries to understand user intent."},
                    {"role": "user", "content": query_analysis_prompt}
                ]
            )
            query_info = json.loads(query_analysis.choices[0].message.content)
        except Exception as e:
            st.error(f"Error analyzing query: {str(e)}")
            query_info = {
                "product_type": query,
                "usage_context": [],
                "requirements": [],
                "related_products": []
            }
        
        # Create a rich query context that includes all extracted information
        query_context = f"""
        Find products that match this query: {query}
        
        Product Type: {query_info['product_type']}
        Usage Context: {', '.join(query_info['usage_context'])}
        Requirements: {', '.join(query_info['requirements'])}
        Related Products: {', '.join(query_info['related_products'])}
        
        Consider:
        - Product name and purpose
        - Product type and category
        - Product description and features
        - Product usage and context
        - Target locations and scenarios
        - Product specifications
        """
        
        # Get query embedding
        query_emb = get_openai_embedding(query_context)
        if not query_emb:
            return []

        # Search in Pinecone with semantic similarity
        index = init_pinecone()
        if not index:
            return []

        # Get more results initially for better filtering
        results = index.query(
            vector=query_emb,
            top_k=top_k * 2,
            include_metadata=True,
            namespace="products"
        )
        
        # Process results with enhanced scoring
        retrieved_products = []
        query_terms = query.lower().split()
        
        for match in results.matches:
            product_id = int(match.id)
            if product_id in st.session_state.products:
                product = st.session_state.products[product_id]
                metadata = match.metadata
                
                # Get product details
                name = product.get('name', '').lower()
                description = metadata.get('description', '').lower()
                category = metadata.get('category', '').lower()
                tags = [tag.lower() for tag in metadata.get('tags', [])]
                
                # Calculate semantic similarity score (from Pinecone)
                semantic_score = match.score
                
                # Calculate name match with context
                name_match = 0.0
                if name:
                    # Check for exact matches with product type
                    if query_info['product_type'].lower() in name:
                        name_match = 1.0
                    # Check for related product matches
                    elif any(related.lower() in name for related in query_info['related_products']):
                        name_match = 0.8
                    # Check for word matches
                    else:
                        matching_words = sum(1 for term in query_terms if term in name)
                        name_match = matching_words / len(query_terms)
                
                # Calculate description match with context
                description_match = 0.0
                if description:
                    # Check for usage context matches
                    context_matches = sum(1 for context in query_info['usage_context'] if context.lower() in description)
                    if context_matches > 0:
                        description_match += 0.4 * (context_matches / len(query_info['usage_context']))
                    
                    # Check for requirement matches
                    req_matches = sum(1 for req in query_info['requirements'] if req.lower() in description)
                    if req_matches > 0:
                        description_match += 0.3 * (req_matches / len(query_info['requirements']))
                    
                    # Check for product type matches
                    if query_info['product_type'].lower() in description:
                        description_match += 0.3
                    # Check for related product matches
                    elif any(related.lower() in description for related in query_info['related_products']):
                        description_match += 0.2
                
                # Calculate category match
                category_match = 0.0
                if category:
                    if query_info['product_type'].lower() in category:
                        category_match = 1.0
                    elif any(related.lower() in category for related in query_info['related_products']):
                        category_match = 0.8
                    else:
                        matching_words = sum(1 for term in query_terms if term in category)
                        category_match = matching_words / len(query_terms)
                
                # Calculate tag match with context
                tag_matches = 0
                for tag in tags:
                    # Check for usage context matches
                    if any(context.lower() in tag for context in query_info['usage_context']):
                        tag_matches += 2
                    # Check for requirement matches
                    if any(req.lower() in tag for req in query_info['requirements']):
                        tag_matches += 1
                    # Check for product type matches
                    if query_info['product_type'].lower() in tag:
                        tag_matches += 2
                    # Check for related product matches
                    if any(related.lower() in tag for related in query_info['related_products']):
                        tag_matches += 1
                
                tag_score = min(1.0, tag_matches / (len(query_terms) * 2))
                
                # Combine scores with emphasis on context and relevance
                final_score = (
                    (semantic_score * 0.3) +           # Semantic similarity
                    (description_match * 0.3) +        # Description match (high weight for context)
                    (name_match * 0.2) +               # Name match
                    (category_match * 0.1) +           # Category match
                    (tag_score * 0.1)                  # Tag match
                )
                
                # Only include products with a minimum score threshold
                if final_score > 0.15:  # Lowered threshold for better recall
                    retrieved_products.append({
                        "product": product,
                        "score": final_score,
                        "semantic_score": semantic_score,
                        "name_match": name_match,
                        "description_match": description_match,
                        "category_match": category_match,
                        "tag_score": tag_score,
                        "context_matches": context_matches if 'context_matches' in locals() else 0,
                        "requirement_matches": req_matches if 'req_matches' in locals() else 0
                    })
        
        # Sort by final score and return top results
        retrieved_products.sort(key=lambda x: x["score"], reverse=True)
        return retrieved_products[:top_k]
        
    except Exception as e:
        st.error(f"Error retrieving products: {str(e)}")
        return []

# --- Initialize session state for updates ---
if 'last_updated_product' not in st.session_state:
    st.session_state.last_updated_product = None

# Initialize session state for form inputs
if 'form_inputs' not in st.session_state:
    st.session_state.form_inputs = {}

def update_form_input(product_id, field, value):
    if product_id not in st.session_state.form_inputs:
        st.session_state.form_inputs[product_id] = {}
    st.session_state.form_inputs[product_id][field] = value

# --- Streamlit Application ---

# Initialize data and models only once when the app starts
if 'initialized' not in st.session_state:
    if not init_data():
        st.error("Failed to initialize application. Please check the console for errors.")
        st.stop()
    st.session_state.initialized = True

# Sidebar: choose between Product Listing and Chatbot
page = st.sidebar.selectbox("Choose Page", ["Product Listing", "Chatbot"])

# ---------- Product Listing Screen ----------
if page == "Product Listing":
    st.title("Product Listing")
    st.write("Below is the list of products. You can update product details or resync embeddings.")
    
    # Add pagination for better performance
    products_per_page = 100  # Increased from 50 to 100
    total_products = len(st.session_state.products)
    total_pages = (total_products + products_per_page - 1) // products_per_page
    
    # Add page selector with more options
    col1, col2 = st.columns([3, 1])
    with col1:
        current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    with col2:
        st.write(f"Total Products: {total_products}")
    
    # Calculate start and end indices for current page
    start_idx = (current_page - 1) * products_per_page
    end_idx = min(start_idx + products_per_page, total_products)
    
    # Show current page info
    st.info(f"Showing products {start_idx + 1} to {end_idx} of {total_products} total products")
    
    # Add search and filter options
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("Search by name or category:")
    with col2:
        sort_by = st.selectbox("Sort by:", ["ID", "Name", "Category"])
    
    # Create a copy of products for the current page
    current_products = dict(list(st.session_state.products.items())[start_idx:end_idx])
    
    # Apply search filter if provided
    if search_term:
        search_term = search_term.lower()
        current_products = {
            k: v for k, v in current_products.items()
            if search_term in v.get('name', '').lower() or search_term in v.get('category', 'Uncategorized').lower()
        }
    
    # Sort products if needed
    if sort_by == "Name":
        current_products = dict(sorted(current_products.items(), key=lambda x: x[1].get('name', '')))
    elif sort_by == "Category":
        current_products = dict(sorted(current_products.items(), key=lambda x: x[1].get('category', 'Uncategorized')))
    
    # Display products for current page
    for product_id, product in current_products.items():
        expander_label = f"ID {product['id']} - {product['name']}"
        with st.expander(expander_label):
            # Display current product details
            st.write("**Current Details:**")
            st.write(f"**Name:** {product.get('name', '')}")
            st.write(f"**Slug:** {product.get('slug', '')}")
            st.write(f"**Description:** {clean_description(product.get('description', ''))}")
            st.write(f"**Category:** {product.get('category', 'Uncategorized')}")
            st.write(f"**Tags:** {', '.join(product.get('tags', []))}")
            st.write(f"**SKU:** {product.get('sku', 'N/A')}")
            
            # Update form
            st.write("---")
            st.write("**Update Product Details:**")
            new_name = st.text_input("New Name", value=product.get('name', ''), key=f"name_{product.get('id', '')}")
            new_desc = st.text_area("New Description", value=clean_description(product.get('description', '')), key=f"desc_{product.get('id', '')}")
            new_category = st.text_input("New Category", value=product.get('category', 'Uncategorized'), key=f"category_{product.get('id', '')}")
            new_tags = st.text_input("New Tags (comma-separated)", value=', '.join(product.get('tags', [])), key=f"tags_{product.get('id', '')}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Update Details", key=f"update_{product.get('id', '')}"):
                    try:
                        # Create updated product
                        updated_product = {
                            "id": product.get('id', ''),
                            "name": new_name,
                            "slug": product.get('slug', ''),  # Keep original slug
                            "description": new_desc,
                            "category": new_category,
                            "tags": [tag.strip() for tag in new_tags.split(',') if tag.strip()],
                            "sku": product.get('sku', f"SKU_{product.get('id', '')}")
                        }
                        
                        # Update product in session state
                        st.session_state.products[product.get('id', '')] = updated_product
                        st.session_state.last_updated_product = product.get('id', '')
                        
                        # Save to JSON file
                        if not save_services_data(st.session_state.products):
                            st.error("Failed to save product updates!")
                            st.stop()
                        
                        # Update embedding and index immediately
                        if update_product_embedding(updated_product):
                            st.success("Product details updated and embeddings re-synced successfully!")
                            # Force a rerun to refresh the UI
                            st.rerun()
                        else:
                            st.error("Failed to update product embedding! Please try again.")
                    except Exception as e:
                        st.error(f"Error updating product: {str(e)}")
            
            with col2:
                if st.button("Resync Embedding", 
                           key=f"resync_{product.get('id', '')}",
                           disabled=(st.session_state.last_updated_product != product.get('id', ''))):
                    try:
                        current_product = st.session_state.products[product.get('id', '')]
                        if update_product_embedding(current_product):
                            st.success("Embedding re-synced successfully!")
                            # Force a rerun to refresh the UI
                            st.rerun()
                        else:
                            st.error("Failed to update product embedding! Please try again.")
                        st.session_state.last_updated_product = None
                    except Exception as e:
                        st.error(f"Error resyncing embedding: {str(e)}")
    
    # Add navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if current_page > 1:
            if st.button("Previous Page"):
                st.session_state.current_page = current_page - 1
                st.rerun()
    with col2:
        st.write(f"Page {current_page} of {total_pages}")
    with col3:
        if current_page < total_pages:
            if st.button("Next Page"):
                st.session_state.current_page = current_page + 1
                st.rerun()

# ---------- Chatbot Screen ----------
elif page == "Chatbot":
    st.title("Product Search Chatbot")
    st.write("Enter a product query. The system will find relevant products based on semantic similarity, categories, and tags.")
    
    query = st.text_input("Enter your product query:")
    if st.button("Search"):
        if query.strip() == "":
            st.warning("Please enter a query!")
        else:
            # Process the query
            processed_query = process_query(query)
            
            # Get relevant products (top 20 instead of 100)
            relevant_products = get_relevant_products(processed_query, top_k=20)
            
            if relevant_products:
                # Display results
                results = []
                for item in relevant_products:
                    product = item["product"]
                    description = clean_description(product.get('description', ''))
                    results.append({
                        "ID": product.get('id', ''),
                        "Product Name": product.get('name', ''),
                        "Description": description,
                        "Category": product.get('category', 'Uncategorized'),
                        "Tags": ', '.join(product.get('tags', [])),
                        "Overall Score": round(item["score"], 4),
                        "Semantic Score": round(item["semantic_score"], 4),
                        "Description Match": round(item["description_match"], 4)
                    })
                
                result_df = pd.DataFrame(results)
                st.write(f"### Retrieved Products (Top {len(results)}):")
                st.dataframe(result_df)
                st.info("Note: Use the product ID to find and update the product in the Product Listing section.")
            else:
                st.warning("No relevant products found for your query.")
