"""
Queue Manager for the Groq Cloud API client.

This module provides a priority-based asynchronous request queue that respects rate limits
and executes requests according to priority levels.
"""
import asyncio
import uuid
from typing import Any, List, Dict, Optional, Callable, Awaitable, Union

from .api_client import APIClient
from .rate_limit_handler import RateLimitHandler


class QueueManager:
    """
    A priority-based queue manager for API requests that respects rate limits.
    
    The QueueManager maintains three priority queues (high, normal, low) and processes
    requests according to their priority level and the current rate limits. It handles
    request scheduling, execution, and callback invocation.
    """
    def __init__(self, api_client: APIClient, rate_limit_handler: RateLimitHandler) -> None:
        """
        Initializes the QueueManager with an API client and a rate limit handler.

        Args:
            api_client (APIClient): The API client to use for making requests.
            rate_limit_handler (RateLimitHandler): The rate limit handler to use for managing rate limits.
        """
        self.api_client = api_client                                                                                    # Set the API client
        self.rate_limit_handler = rate_limit_handler                                                                    # Set the rate limit handler

        # Create priority queues for each priority level
        self.high_priority_queue: List[Dict[str, Any]] = []                                                             # Create a high priority queue
        self.normal_priority_queue: List[Dict[str, Any]] = []                                                           # Create a normal priority queue
        self.low_priority_queue: List[Dict[str, Any]] = []      # default queue                                         # Create a low priority queue

        self.processing_task: Optional[asyncio.Task] = None                                                             # Task for processing the queue
        self.running = False                                                                                            # Flag to indicate if the queue manager is running
        self.queue_lock = asyncio.Lock()                                                                                # Create a lock for the queue
        self.request_map: Dict[str, Dict[str, Any]] = {}                                                                # Map of request IDs to requests

    def start(self) -> None:
        """
        Starts the queue manager to process the queues.

        This method safely starts the queue manager both inside
        and outside of the event loop.
        """
        if not self.running:                                                                                            # Check if the queue manager is already running
            self.running = True                                                                                         # Set the running flag to True
            # Store the fact that the queue manager is running
            # The task will be created in the event loop

    def stop(self) -> None:
        """
        Stops the queue manager and cancels any running tasks.
        """
        if self.running:                                                                                                # Check if the queue manager is running
            self.running = False                                                                                        # Set the running flag to False
            if self.processing_task is not None:                                                                        # Check if there is a processing task
                self.processing_task.cancel()                                                                           # Cancel the processing task
                self.processing_task = None                                                                             # Set the processing task to None

    async def ensure_processing(self) -> None:
        """
        Ensure the processing loop is running in the current event loop.
        This method is called automatically by enqueue_request.
        """
        if self.running and (self.processing_task is None or self.processing_task.done()):                              # Check if the queue manager is running and if the processing task is None or done
            self.processing_task = asyncio.create_task(self._process_queue())                                           # Create a new processing task

    async def enqueue_request(self,
                              endpoint: str,
                              payload: Dict[str, Any],
                              token_count: int,
                              callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
                              priority: str = "low"  # Default priority is low
                              ) -> str:
        """
        Enqueue a request to be processed according to rate limits and priority.

        Args:
            endpoint: API endpoint to call
            payload: Request payload
            token_count: Estimated token count for the request
            callback: Optional async callback function to be called with the API response
            priority: Priority level - "high", "normal", or "low" (default is "low")
            
        Returns:
            str: Unique ID for this request, can be used to cancel it later
        """
        request_id = str(uuid.uuid4())                                                                                  # Create unique id for reuest
        
        request = {                                                                                                     # Create result
            "id": request_id,
            "endpoint": endpoint,                                                                                       # API endpoint
            "payload": payload,                                                                                         # Request payload
            "token_count": token_count,                                                                                 # Estimated token count
            "callback": callback,                                                                                       # Optional callback function
            "enqueue_time": asyncio.get_event_loop().time(),                                                            # Enqueue time
            "priority": priority                                                                                        # Priority level
        }

        async with self.queue_lock:
            # Add the request to the appropriate priority queue
            if priority == "high":                                                                                      # Check if the priority is high
                self.high_priority_queue.append(request)                                                                # Add to high priority queue
            elif priority == "normal":                                                                                  # Check if the priority is normal
                self.normal_priority_queue.append(request)                                                              # Add to medium priority queue
            elif priority == "low":                                                                                     # Check if the priority is low
                self.low_priority_queue.append(request)                                                                 # Add to low priority queue
            else:
                raise ValueError(f"Invalid priority level: {priority}")                                                 # Raise an error if the priority is invalid
                
            # Store the request in the request map
            self.request_map[request_id] = request

        # Ensure the processing loop is running
        await self.ensure_processing()                                                                                  # Call the ensure_processing method
        
        return request_id

    def get_next_request(self) -> Optional[Dict[str, Any]]:
        """
        Get the next request to process based on priority.

        Returns:
            Dict[str, Any]: The next request to process, or None if no requests are available
        """
        if self.high_priority_queue:                                                                                    # Check if there are requests in the high priority queue
            return self.high_priority_queue.pop(0)                                                                      # Get the first request from the high priority queue
        elif self.normal_priority_queue:                                                                                # Check if there are requests in the normal priority queue
            return self.normal_priority_queue.pop(0)                                                                    # Get the first request from the normal priority queue
        elif self.low_priority_queue:                                                                                   # Check if there are requests in the low priority queue
            return self.low_priority_queue.pop(0)                                                                       # Get the first request from the low priority queue
        return None                                                                                                     # Return None if no requests are available

    def get_queue_length(self) -> Dict[str, int]:
        """
        Get the length of all queues.

        Returns:
            Dict[str, int]: Dictionary with queue lengths by priority level
        """
        return {                                                                                                        # Get the length of all queues
            "high": len(self.high_priority_queue),                                                                      # Length of high priority queue
            "normal": len(self.normal_priority_queue),                                                                  # Length of normal priority queue
            "low": len(self.low_priority_queue),                                                                        # Length of low priority queue
            "total": len(self.high_priority_queue) + len(self.normal_priority_queue) + len(self.low_priority_queue)     # Total length of all queues
        }

    async def _process_queue(self) -> None:
        """
        Process the queue in order and handle rate limits.
        """
        while self.running:
            request = None
            
            try:
                async with self.queue_lock:
                    # Check if there are requests to process in the queue lists
                    if not (self.high_priority_queue or self.normal_priority_queue or self.low_priority_queue):
                        continue  # No requests to process, continue to next iteration
                        
                    # Get the next request to process (by priority)
                    request = self.get_next_request()
                    if request is None:
                        continue  # No requests to process, continue to next iteration
                        
                    # Check if we can process the request based on rate limits
                    can_process, reasons = self.rate_limit_handler.can_make_request(request["token_count"])
                    
                    if not can_process:
                        # Cannot process the request due to rate limits, requeue it
                        priority = request.get("priority", "low")
                        if priority == "high":
                            self.high_priority_queue.insert(0, request)
                        elif priority == "normal":
                            self.normal_priority_queue.insert(0, request)
                        else:
                            self.low_priority_queue.insert(0, request)
                            
                        # Wait before checking again
                        continue
                        
                    # Remove from request map before processing
                    request_id = request["id"]
                    if request_id in self.request_map:
                        del self.request_map[request_id]
            
                # Don't hold the lock while we process the request
                if request:
                    await self._process_request(request)
            
            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                break
            except Exception as e:
                # Error during queue processing, log and continue
                print(f"Error in queue processing: {e}")
                
            # Sleep a short time before checking again
            await asyncio.sleep(0.1)
                
    async def _process_request(self, request: Dict[str, Any]) -> None:
        """
        Process a single request and invoke its callback.
        
        Args:
            request: The request to process
        """
        try:
            # Send the request
            response = await self._send_request(request["endpoint"], request["payload"])
            
            # Update rate limit counters
            self.rate_limit_handler.update_counters(request["token_count"])
            
            # Call the callback function if provided
            if request["callback"] is not None:
                try:
                    await request["callback"](response)
                except Exception as callback_e:
                    print(f"Error in callback for request {request['id']}: {callback_e}")
                    
        except Exception as e:
            print(f"Error processing request {request['id']}: {e}")
            
            # If there's a callback, call it with the error
            if request["callback"]:
                try:
                    await request["callback"]({"error": str(e)})
                except Exception as callback_e:
                    print(f"Error in error callback for request {request['id']}: {callback_e}")

    async def _send_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request using the API client.

        Args:
            endpoint: API endpoint to call
            payload: Request payload

        Returns:
            Dict[str, Any]: API response
        """
        return await self.api_client.post_request(endpoint, payload)

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the current status of the request queue.

        Returns:
            Dict[str, Any]: Dictionary with queue status information
        """
        queue_lengths = self.get_queue_length()
        return {
            "queue_lengths": queue_lengths,
            "total_queue_length": queue_lengths["total"],
            "running": self.running,
            "rate_limits": self.rate_limit_handler.get_status()
        }

    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a pending request by ID.

        Args:
            request_id: The ID of the request to cancel

        Returns:
            bool: True if the request was found and cancelled, False otherwise
        """
        async with self.queue_lock:
            # Check if the request is in the request map
            if request_id not in self.request_map:
                return False
                
            request = self.request_map[request_id]
            priority = request.get("priority", "low")
            
            # Remove the request from the appropriate queue
            if priority == "high":
                if request in self.high_priority_queue:
                    self.high_priority_queue.remove(request)
            elif priority == "normal":
                if request in self.normal_priority_queue:
                    self.normal_priority_queue.remove(request)
            else:
                if request in self.low_priority_queue:
                    self.low_priority_queue.remove(request)
                    
            # Remove from the request map
            del self.request_map[request_id]
            return True