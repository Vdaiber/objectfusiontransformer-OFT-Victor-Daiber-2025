"""
Training metrics logging and monitoring utilities.

This module provides utility classes for tracking, smoothing, and formatting
metrics during machine learning model training. The implementation includes
statistical smoothing techniques and progress monitoring capabilities.

Key components:
- SmoothedValue: Tracks moving average/median statistics for metric values
- MetricLogger: Manages collections of smoothed metrics and provides
  formatted progress reporting at regular intervals

These implementations are inspired by common PyTorch example repositories
and provide robust metric tracking for training pipelines.
"""

import torch
import time
import datetime
import logging
from collections import deque, defaultdict

class SmoothedValue(object):
    """Statistical smoothing utility for tracking metric values over time.
    
    Maintains a sliding window of values and provides various statistical
    measures including median, mean, and global average. The smoothing helps
    reduce noise in training metrics and provides more stable progress indicators.
    """
    
    def __init__(self, window_size=20, fmt=None):
        """Initialize SmoothedValue with specified window size and format.
        
        Args:
            window_size: Number of recent values to maintain in sliding window.
            fmt: Format string for string representation of statistics.
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """Add new value(s) to the tracked sequence.
        
        Args:
            value: New value to add to the sequence.
            n: Number of samples this value represents (for weighted averaging).
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        """Calculate median of values in the current sliding window.
        
        Returns:
            Median value of the recent window, or 0.0 if window is empty.
        """
        d = torch.tensor(list(self.deque))
        return d.median().item() if len(d) > 0 else 0.0

    @property
    def avg(self):
        """Calculate mean of values in the current sliding window.
        
        Returns:
            Mean value of the recent window, or 0.0 if window is empty.
        """
        d = torch.tensor(list(self.deque), dtype=torch.float64)
        return d.mean().item() if len(d) > 0 else 0.0

    @property
    def global_avg(self):
        """Calculate global average of all values seen since initialization.
        
        Returns:
            Global average across all updates, or 0.0 if no updates.
        """
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def max(self):
        """Find maximum value in the current sliding window.
        
        Returns:
            Maximum value in recent window, or 0.0 if window is empty.
        """
        return max(self.deque) if len(self.deque) > 0 else 0.0

    @property
    def value(self):
        """Get the most recently added value.
        
        Returns:
            Last value added to the sequence, or 0.0 if no values.
        """
        return self.deque[-1] if len(self.deque) > 0 else 0.0

    def __str__(self):
        """String representation of current statistics.
        
        Returns:
            Formatted string showing median and global average statistics.
        """
        if self.count == 0:
            return "N/A"
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


class MetricLogger(object):
    """Comprehensive metric tracking and logging system for training pipelines.
    
    Manages collections of SmoothedValue metrics and provides formatted progress
    reporting at regular intervals. Includes timing statistics, memory usage
    monitoring, and estimated completion times.
    
    The logger can wrap any iterable (e.g., DataLoader) to automatically
    log progress during training iterations.
    """
    
    def __init__(self, delimiter="\t", logger=None):
        """Initialize MetricLogger with specified formatting options.
        
        Args:
            delimiter: String separator for formatting metric output.
            logger: Logger instance for output (creates default if None).
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger if logger else logging.getLogger("default_metric_logger")

    def update(self, **kwargs):
        """Update one or more metrics with new values.
        
        Args:
            **kwargs: Keyword arguments mapping metric names to values.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def add_meter(self, name: str, meter: SmoothedValue):
        """Add an external SmoothedValue object to the metric collection.
        
        Args:
            name: Name identifier for the metric.
            meter: SmoothedValue instance to add.
        """
        self.meters[name] = meter

    def __getattr__(self, attr):
        """Enable direct access to metrics as attributes.
        
        Args:
            attr: Attribute name (metric name).
            
        Returns:
            SmoothedValue instance for the requested metric.
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        """String representation of all tracked metrics.
        
        Returns:
            Formatted string showing all metric names and their current statistics.
        """
        return self.delimiter.join(f"{name}: {str(meter)}" for name, meter in self.meters.items())

    def log_every(self, iterable, print_freq, header=None):
        """Iterator wrapper that logs progress at regular intervals.
        
        This method wraps any iterable (e.g., DataLoader) and automatically
        logs progress information including timing, metrics, and estimated
        completion time at specified intervals.
        
        Args:
            iterable: Iterable object to wrap (e.g., DataLoader)
            print_freq: Frequency of logging (log every N iterations)
            header: Optional header string for log messages
            
        Yields:
            Items from the original iterable
        """
        i = 0
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        try:
            iterable_len = len(iterable)
        except TypeError:
            iterable_len = -1
        
        # Format string for iteration counter
        space_fmt = f':{str(len(str(iterable_len)))}d' if iterable_len > 0 else ''
        log_msg_parts = [
            header if header else '',
            '[{0' + space_fmt + '}/{1}]' if iterable_len > 0 else '[{0' + space_fmt + '}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg_parts.append('max mem: {memory:.0f}')
        
        log_msg = self.delimiter.join(log_msg_parts)
        MB = 1024.0 * 1024.0
        
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            
            current_len_for_log = iterable_len if iterable_len > 0 else i + 1
            if i % print_freq == 0 or (iterable_len > 0 and i == iterable_len - 1):
                # Calculate estimated time to completion
                eta_seconds = iter_time.global_avg * (current_len_for_log - 1 - i) if iterable_len > 0 and iter_time.count > 0 else 0
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                # Prepare format dictionary for logging
                format_dict = {
                    "eta": eta_string,
                    "meters": str(self),
                    "time": str(iter_time),
                    "data": str(data_time)
                }
                if torch.cuda.is_available():
                    format_dict["memory"] = torch.cuda.max_memory_allocated() / MB
                
                log_format_args = [i, current_len_for_log] if iterable_len > 0 else [i]
                
                if self.logger:
                    self.logger.info(log_msg.format(*log_format_args, **format_dict))
            
            i += 1
            end = time.time()
        
        # Log final summary statistics
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        avg_time_per_it = total_time / i if i > 0 else 0
        
        if self.logger:
            self.logger.info(f'{header if header else ""} Total time: {total_time_str} ({avg_time_per_it:.4f} s / it)') 