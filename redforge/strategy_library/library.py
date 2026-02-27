"""Strategy library manager using ChromaDB as vector store backend."""

import os
import logging
from typing import Optional

import chromadb

from redforge.strategy_library.entry import StrategyEntry

logger = logging.getLogger(__name__)


class StrategyLibrary:
    """Manages the persistent strategy library using ChromaDB."""

    def __init__(self, persist_dir: str = "data/strategy_library"):
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="redforge_strategies",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, entry: StrategyEntry) -> str:
        """Add a strategy entry to the library.

        Returns:
            The strategy_id
        """
        self._collection.upsert(
            ids=[entry.strategy_id],
            documents=[entry.get_embedding_text()],
            metadatas=[entry.to_dict()],
        )
        logger.info(f"Added strategy: {entry.technique_name} ({entry.strategy_id})")
        return entry.strategy_id

    def get(self, strategy_id: str) -> Optional[StrategyEntry]:
        """Get a strategy entry by ID."""
        results = self._collection.get(ids=[strategy_id], include=["metadatas"])
        if results["metadatas"]:
            return StrategyEntry.from_dict(results["metadatas"][0])
        return None

    def update(self, entry: StrategyEntry) -> None:
        """Update an existing strategy entry."""
        self._collection.upsert(
            ids=[entry.strategy_id],
            documents=[entry.get_embedding_text()],
            metadatas=[entry.to_dict()],
        )

    def search(
        self,
        query_text: str,
        n_results: int = 5,
        category: Optional[str] = None,
    ) -> list[tuple[StrategyEntry, float]]:
        """Search for similar strategies.

        Args:
            query_text: Text to search for
            n_results: Number of results to return
            category: Optional category filter

        Returns:
            List of (StrategyEntry, similarity_score) tuples
        """
        where_filter = None
        if category:
            where_filter = {"category": category}

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter,
                include=["metadatas", "distances"],
            )
        except Exception:
            # ChromaDB may raise if collection is empty
            return []

        entries = []
        if results["metadatas"] and results["metadatas"][0]:
            for metadata, distance in zip(
                results["metadatas"][0], results["distances"][0]
            ):
                entry = StrategyEntry.from_dict(metadata)
                # ChromaDB cosine distance is 1 - similarity
                similarity = 1.0 - distance
                entries.append((entry, similarity))

        return entries

    def find_similar(
        self, entry: StrategyEntry, threshold: float = 0.7
    ) -> Optional[tuple[StrategyEntry, float]]:
        """Find the most similar existing strategy.

        Returns:
            (matching_entry, similarity) if similarity >= threshold, else None
        """
        results = self.search(entry.get_embedding_text(), n_results=1)
        if results:
            match, similarity = results[0]
            if similarity >= threshold and match.strategy_id != entry.strategy_id:
                return match, similarity
        return None

    def count(self) -> int:
        """Get the total number of strategies in the library."""
        return self._collection.count()

    def list_all(self, limit: int = 100) -> list[StrategyEntry]:
        """List all strategies in the library."""
        results = self._collection.get(
            include=["metadatas"],
            limit=limit,
        )
        return [StrategyEntry.from_dict(m) for m in results["metadatas"]]

    def get_stats(self) -> dict:
        """Get library statistics."""
        entries = self.list_all(limit=10000)
        if not entries:
            return {"total": 0}

        categories = {}
        total_success_rate = 0.0
        novel_count = 0

        for entry in entries:
            cat = entry.category
            if cat not in categories:
                categories[cat] = {"count": 0, "avg_success_rate": 0.0}
            categories[cat]["count"] += 1
            categories[cat]["avg_success_rate"] += entry.success_rate
            total_success_rate += entry.success_rate
            if entry.novelty_score > 0.7:
                novel_count += 1

        for cat in categories:
            n = categories[cat]["count"]
            categories[cat]["avg_success_rate"] /= n if n else 1

        return {
            "total": len(entries),
            "categories": categories,
            "avg_success_rate": total_success_rate / len(entries),
            "novel_strategies": novel_count,
        }
