import re
import json
from typing import Optional
from typing import Any, List, Dict, Optional, Tuple, cast
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
)
import asyncio
from .utils import (
    logger,
    compute_mdhash_id,
    handle_cache,
    save_to_cache,
    CacheData,
)

class MultiModalProcessor:
    def __init__(
        self,
        modal_caption_func,  # 多模态内容生成caption的函数
        text_chunks_db: BaseKVStorage,
        chunks_vdb: BaseVectorStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        knowledge_graph_inst: BaseGraphStorage,
        embedding_func,
        llm_model_func,
        global_config: dict,
        hashing_kv: Optional[BaseKVStorage] = None,
    ):
        self.modal_caption_func = modal_caption_func
        self.text_chunks_db = text_chunks_db
        self.chunks_vdb = chunks_vdb
        self.entities_vdb = entities_vdb
        self.relationships_vdb = relationships_vdb
        self.knowledge_graph_inst = knowledge_graph_inst
        self.embedding_func = embedding_func
        self.llm_model_func = llm_model_func
        self.global_config = global_config
        self.hashing_kv = hashing_kv

    async def process_multimodal_content(
        self,
        modal_content,  # 多模态内容（图像、视频等）
        content_type: str,  # 内容类型，如"image", "video"等
        entity_name: str = None,  # 可选的实体名称
        top_k: int = 10,
        better_than_threshold: float = 0.6,
        file_path: str = "unkown_source",
    ) -> Tuple[str, Dict[str, Any]]:
        """Process multimodal content and create a single entity in the knowledge graph"""
        
        # 1. Generate initial caption
        initial_caption = await self._generate_initial_caption(modal_content, content_type)
        logger.debug(f"Initial caption generated: {initial_caption}")
        
        # 2. Retrieve related text chunks
        related_chunks = await self._retrieve_related_chunks(initial_caption, top_k, better_than_threshold)
        
        # 3. Generate enhanced caption
        enhanced_caption = await self._generate_enhanced_caption(initial_caption, related_chunks)
        logger.debug(f"Enhanced caption generated: {enhanced_caption}")
        
        # 4. Create entity ID if not provided
        if not entity_name:
            entity_name = f"{content_type}_{compute_mdhash_id(str(modal_content))}"
        
        # 5. Update knowledge graph with the single entity
        source_id = compute_mdhash_id(str(modal_content))
        
        # Create node data for the entity
        node_data = {
            "entity_id": entity_name,
            "entity_type": content_type,  # Use content type as entity type
            "description": enhanced_caption,
            "source_id": source_id,
            "file_path": file_path,  # Optional, can be customized
        }
        # Add entity to knowledge graph
        await self.knowledge_graph_inst.upsert_node(entity_name, node_data)
        
        # Insert entity into entities vector database
        entity_vdb_data = {
            compute_mdhash_id(entity_name, prefix="ent-"): {
                "entity_name": entity_name,
                "entity_type": content_type,
                "content": f"{entity_name}\n{enhanced_caption}",
                "source_id": source_id,
                "file_path": file_path,
            }
        }
        await self.entities_vdb.upsert(entity_vdb_data)
        
        # 6. Find potential relationships with existing entities
        relationships = await self._find_related_entities(node_data)
        
        # 7. Add relationships to knowledge graph
        for relation in relationships:
            edge_data = {
                "weight": relation["weight"],
                "keywords": relation["keywords"],
                "description": relation["description"],
                "source_id": source_id,
                "file_path": file_path,
            }
            await self.knowledge_graph_inst.upsert_edge(relation["source"], relation["target"], edge_data)
            
            # Insert relationship into relationships vector database
            relationship_vdb_data = {
                compute_mdhash_id(relation["source"] + relation["target"], prefix="rel-"): {
                    "src_id": relation["source"],
                    "tgt_id": relation["target"],
                    "keywords": relation["keywords"],
                    "content": f"{relation['source']}\t{relation['target']}\n{relation['keywords']}\n{relation['description']}",
                    "source_id": source_id,
                    "file_path": file_path,
                }
            }
            await self.relationships_vdb.upsert(relationship_vdb_data)

        await self._insert_done()

        return enhanced_caption, {
            "entity_name": entity_name, 
            "entity_type": content_type,
            "description": enhanced_caption,
            "relationships": relationships
        }

    async def _insert_done(self) -> None:
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [  # type: ignore
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.knowledge_graph_inst,
                    self.hashing_kv,
                ]
            ]
        )

    async def _generate_initial_caption(self, modal_content, content_type: str) -> str:
        """Generate initial caption using modality-specific LLM"""
        # Cache handling
        content_hash = compute_mdhash_id(str(modal_content))
        args_hash = f"modal_caption_{content_hash}"
        
        cached_result, quantized, min_val, max_val = await handle_cache(
            self.hashing_kv, args_hash, str(content_type), mode="modal_caption", cache_type="modal"
        )
        
        if cached_result:
            return cached_result
        
        # 构建提示(prompt)
        caption_prompt = self._build_caption_prompt(content_type)
        
        # Generate caption
        caption = await self.modal_caption_func(modal_content, system_prompt=caption_prompt)
        
        # Save to cache
        if self.hashing_kv:
            await save_to_cache(
                self.hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=caption,
                    prompt=caption_prompt,
                    quantized=quantized,
                    min_val=min_val,
                    max_val=max_val,
                    mode="modal_caption",
                    cache_type="modal",
                ),
            )

        return caption
        
    def _build_caption_prompt(self, content_type: str) -> str:
        """构建用于生成caption的提示"""
        base_prompt = "Please provide a detailed and comprehensive description of this content in English only. "
        
        type_specific_prompts = {
            "image": "Describe what you see in this image. Include all visible objects, people, actions, colors, text, environment, and spatial relationships. Be specific and objective.",
            "video": "Describe what happens in this video. Include key events, scenes, people, actions, objects, and any relevant audio or text content. Provide a chronological overview.",
            "audio": "Describe this audio content. Include the type of audio (speech, music, sound effects), speakers if any, key topics discussed, mood, tone, and notable audio elements.",
            "markdown_table": "Describe this markdown table. Include the table type, structure, key topics, important text content, and any visible formatting or visual elements.",
        }
        
        specific_prompt = type_specific_prompts.get(content_type.lower(), "Focus on the key features and elements of this content.")
        
        return f"{base_prompt}{specific_prompt} Ensure your description is entirely in English, detailed, and factual."

    async def _retrieve_related_chunks(self, caption: str, top_k: int, better_than_threshold: float) -> List[Dict[str, Any]]:
        """Retrieve relevant text chunks based on caption"""
        # Use vector database to retrieve related text chunks
        results = await self.chunks_vdb.query(
            caption, top_k=top_k,
        )
        
        if not results:
            return None

        return results


    async def _generate_enhanced_caption(self, initial_caption: str, related_chunks: List[Dict[str, Any]]) -> str:
        """Combine original text and initial caption to generate enhanced caption"""
        # Build prompt
        if related_chunks:
            chunks_text = "\n\n".join([chunk["content"] for chunk in related_chunks])
            prompt = f"""Based on the following information, generate a detailed description:
            
            Initial description:
            {initial_caption}

            Related text content:
            {chunks_text}

            Please generate a more detailed and accurate description by combining the initial description with related text content. 
            The description should include key entities, relationships, and important details. Limit the description to approximately 500 words."""
        else:
            prompt = f"""Based on the following information, generate a detailed description:
            
            Initial description:
            {initial_caption}

            Please generate a more detailed and accurate description
            The description should include key entities, relationships, and important details. Limit the description to approximately 500 words.
            """
        # Use LLM to generate enhanced caption
        enhanced_caption = await self.llm_model_func(prompt)
        return enhanced_caption


    async def _find_related_entities(self, node_data: dict, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find relationships between this entity and existing entities in the knowledge graph using vector similarity"""
        
        # 1. Use entity vector database to find related entities based on enhanced caption
        
        # Query the entity vector database using the enhanced caption
        related_entities = await self.entities_vdb.query(
            node_data["description"], 
            top_k=top_k,
        )
        
        if not related_entities:
            return []
        
        # 2. Filter out the current entity if it already exists
        related_entities = [e for e in related_entities if e["entity_name"] != node_data["entity_id"]]
        
        if not related_entities:
            return []
        
        # 3. Get full entity information from knowledge graph
        entity_names = [entity["entity_name"] for entity in related_entities]
        
        # Build context for each related entity
        entity_contexts = []
        entity_data = {}
        
        for ent_name in entity_names:
            # Get node data from knowledge graph
            related_node_data = await self.knowledge_graph_inst.get_node(ent_name)
            if related_node_data:
                entity_data[ent_name] = related_node_data
                ent_desc = related_node_data.get("description", "")
                if ent_desc:
                    entity_contexts.append(f"Entity: {ent_name}\nDescription: {ent_desc}")
        
        if not entity_contexts:
            return []
        
        entity_contexts_str = "\n\n".join(entity_contexts)
        # 4. Build prompt to find relationships using LLM
        prompt = f"""Analyze potential relationships between the new entity and existing entities.

        New entity: {node_data["entity_id"]}
        New entity description: {node_data["description"]}

        Existing entities:
        {entity_contexts_str}

        For each potential relationship between the new entity and an existing entity, provide:
        1. {node_data["entity_id"]}
        2. Target entity name
        3. Relationship description (detailed explanation of how the entities are related)
        4. Relationship keywords (core concepts of the relationship)
        5. Relationship strength (a value between 0.0 and 1.0 indicating strength)

        Only create relationships where there's a clear connection.
        Return results in JSON format:
        [
        {{
            "source": "{node_data["entity_id"]}",
            "target": "target entity name",
            "description": "relationship description",
            "keywords": "relationship keywords",
            "weight": "relationship strength"
        }}
        ]"""
        
        # 5. Use LLM to find relationships
        relationships_json = await self.llm_model_func(prompt)
        # 6. Parse JSON result
        relationships = re.search(r"\{.*\}", relationships_json, re.DOTALL)
        relationships = json.loads(f"[{relationships.group(0)}]")

        return relationships
