"""Provider metrics tracking and statistics."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProviderAttempt:
    """Record of a single provider API attempt."""
    provider_name: str
    timestamp: datetime
    success: bool
    duration: float  # seconds
    cost: float  # USD
    error_message: Optional[str] = None


@dataclass
class ProviderStats:
    """Statistics for a single provider."""
    provider_name: str
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    total_cost: float = 0.0
    total_duration: float = 0.0
    average_duration: float = 0.0
    success_rate: float = 0.0
    attempts_list: List[ProviderAttempt] = field(default_factory=list)
    
    def add_attempt(self, attempt: ProviderAttempt) -> None:
        """Add an attempt record and update statistics."""
        self.attempts += 1
        self.total_duration += attempt.duration
        self.total_cost += attempt.cost
        self.attempts_list.append(attempt)
        
        if attempt.success:
            self.successes += 1
        else:
            self.failures += 1
        
        # Update calculated fields
        if self.attempts > 0:
            self.average_duration = self.total_duration / self.attempts
            self.success_rate = (self.successes / self.attempts) * 100.0
    
    def to_dict(self) -> Dict:
        """Convert stats to dictionary for logging/reporting."""
        return {
            "provider_name": self.provider_name,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": f"{self.success_rate:.1f}%",
            "average_duration": f"{self.average_duration:.2f}s",
            "total_cost": f"${self.total_cost:.4f}",
            "total_duration": f"{self.total_duration:.2f}s"
        }


class ProviderMetrics:
    """Track metrics across all providers."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self._stats: Dict[str, ProviderStats] = {}
        self._session_start = datetime.now()
        self._total_images_processed = 0
        self._total_images_failed = 0
        self._total_images_skipped = 0
    
    def record_attempt(
        self,
        provider_name: str,
        success: bool,
        duration: float,
        cost: float,
        error_message: Optional[str] = None
    ) -> None:
        """
        Record a provider API attempt.
        
        Args:
            provider_name: Name of the provider
            success: Whether the attempt succeeded
            duration: Duration in seconds
            cost: Cost in USD
            error_message: Error message if failed
        """
        if provider_name not in self._stats:
            self._stats[provider_name] = ProviderStats(provider_name=provider_name)
        
        attempt = ProviderAttempt(
            provider_name=provider_name,
            timestamp=datetime.now(),
            success=success,
            duration=duration,
            cost=cost,
            error_message=error_message
        )
        
        self._stats[provider_name].add_attempt(attempt)
        logger.debug(
            f"Recorded {provider_name} attempt: success={success}, "
            f"duration={duration:.2f}s, cost=${cost:.4f}"
        )
    
    def get_stats(self, provider_name: Optional[str] = None) -> Dict:
        """
        Get statistics for a provider or all providers.
        
        Args:
            provider_name: Specific provider name, or None for all
        
        Returns:
            Dictionary with statistics
        """
        if provider_name:
            if provider_name in self._stats:
                return self._stats[provider_name].to_dict()
            return {}
        
        return {
            name: stats.to_dict()
            for name, stats in self._stats.items()
        }
    
    def record_image_processed(self) -> None:
        """Record that an image was successfully processed."""
        self._total_images_processed += 1
    
    def record_image_failed(self) -> None:
        """Record that an image processing failed."""
        self._total_images_failed += 1
    
    def record_image_skipped(self) -> None:
        """Record that an image was skipped."""
        self._total_images_skipped += 1
    
    def get_summary(self) -> Dict:
        """
        Get overall processing summary.
        
        Returns:
            Dictionary with summary statistics
        """
        total_images = (
            self._total_images_processed +
            self._total_images_failed +
            self._total_images_skipped
        )
        
        total_cost = sum(stats.total_cost for stats in self._stats.values())
        total_duration = sum(stats.total_duration for stats in self._stats.values())
        
        session_duration = (datetime.now() - self._session_start).total_seconds()
        
        return {
            "session_start": self._session_start.isoformat(),
            "session_duration": f"{session_duration:.2f}s",
            "total_images": total_images,
            "processed": self._total_images_processed,
            "failed": self._total_images_failed,
            "skipped": self._total_images_skipped,
            "total_cost": f"${total_cost:.4f}",
            "total_duration": f"{total_duration:.2f}s",
            "provider_stats": self.get_stats()
        }
    
    def format_summary_report(self) -> str:
        """
        Format a human-readable summary report.
        
        Returns:
            Multi-line string with formatted report
        """
        summary = self.get_summary()
        lines = []
        
        lines.append("=" * 60)
        lines.append("PROCESSING SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Session Start: {summary['session_start']}")
        lines.append(f"Session Duration: {summary['session_duration']}")
        lines.append("")
        lines.append("IMAGE STATISTICS")
        lines.append("-" * 60)
        lines.append(f"Total Images: {summary['total_images']}")
        lines.append(f"Successfully Processed: {summary['processed']}")
        lines.append(f"Failed: {summary['failed']}")
        lines.append(f"Skipped: {summary['skipped']}")
        lines.append("")
        
        if summary['provider_stats']:
            lines.append("PROVIDER STATISTICS")
            lines.append("-" * 60)
            for provider_name, stats in summary['provider_stats'].items():
                lines.append(f"{provider_name}:")
                lines.append(f"  Attempts: {stats['attempts']}")
                lines.append(f"  Successes: {stats['successes']}")
                lines.append(f"  Failures: {stats['failures']}")
                lines.append(f"  Success Rate: {stats['success_rate']}")
                lines.append(f"  Avg Duration: {stats['average_duration']}")
                lines.append(f"  Total Cost: {stats['total_cost']}")
                lines.append("")
        
        lines.append("COST SUMMARY")
        lines.append("-" * 60)
        lines.append(f"Total Cost: {summary['total_cost']}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all metrics for a new session."""
        self._stats.clear()
        self._session_start = datetime.now()
        self._total_images_processed = 0
        self._total_images_failed = 0
        self._total_images_skipped = 0
        logger.info("Metrics reset for new session")



