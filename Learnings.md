### Pytest
Use pytest.ini to redirect main python path, prevents import errors
 - `testpaths` defines a specific test dir

pytest looks for `test_*.py`, or `*_test.py` and funcs inside with `test_` and treats each as a test

### Asyncio
Use future for one-shot tasks, create_task for streaming tasks that use await/yield

### Python IPC
`queue.Queue` works between Threads and shares mem pointers w/in a process
 - if we multiprocess w/ this queue, we copy it
 - queue's are no longer shared, proc B can't comm w/ proc A over it

`multiprocessing.Queue` is made to work between Processes and handles moving data between processes for us
 - default choice of pickle could explain why VLLM chose to use msgpack + ZMQ sockets here rather than relying on a standard Q

### Python MPC
A process will fail to exit indefinitely if any of it's non-daemon (children) procs are still running
 - all child processes need some kind of signal that lets the parent tell them to exit
 - for example, a child proc that gets data from a Q can have a special Q message that signifies it to exit, sent by it's parent only
