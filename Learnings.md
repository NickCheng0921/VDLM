### Pytest
Use pytest.ini to redirect main python path, prevents import errors
 - `testpaths` defines a specific test dir

pytest looks for `test_*.py`, or `*_test.py` and funcs inside with `test_` and treats each as a test

#### Pytest/TestClient
Why don't we see BE requests when using pytest TestClient?
 - Uses `in-process testing, in-memory, ASGI` testing
 - No OS networking or sockets, ports, TCP/IP but event loop and app is ran
 - Good for fast deterministic API tests
 - Bad for anything that needs network behavior: CORS, deployments

### Asyncio
Use future for one-shot tasks, create_task for streaming tasks that use await/yield

### Python IPC
`queue.Queue` works between Threads and shares mem pointers w/in a process
 - if we multiprocess w/ this queue, we copy it
 - queue's are no longer shared, proc B can't comm w/ proc A over it

`multiprocessing.Queue` is made to work between Processes and handles moving data between processes for us
 - default choice of pickle could explain why VLLM chose to use msgpack + ZMQ sockets here rather than relying on a standard Q

### Python MPC
A process will fail to exit indefinitely if any of its non-daemon (children) procs are still running
 - all child processes need some kind of signal that lets the parent tell them to exit
 - for example, a child proc that gets data from a Q can have a special Q message that signifies it to exit, sent by it's parent only

Child processes should write to a Q in the main thread instead of using own stdout
 - example usage `llm_engine.py`: child clears handlers, attaches new Q as handler, parent creates a LogListener to Q when starting engine child process

Mock engine loops can be kept separate from main loops to remove library loading overhead

### CLI Design
Config flags should lead to immutable instances
 - if engine takes mock cli arg, reinstantiate engine in mock mode