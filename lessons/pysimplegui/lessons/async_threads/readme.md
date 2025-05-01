## Async / Threads in PySimpleGUI

PySimpleGUI provides functionality for handling asynchronous programming and multi-threading to enable the creation of responsive, non-blocking GUI applications. Using threads and asynchronous programming, you can run long-running tasks in the background without freezing the GUI, ensuring smooth interaction with the user.

---

### Key Concepts

1. **Multithreading in PySimpleGUI**:
   - **Multithreading** allows you to run multiple tasks concurrently by creating separate threads, enabling tasks like downloading files, processing data, or running calculations to occur in parallel without blocking the main thread (which is responsible for GUI updates).
   - PySimpleGUI integrates well with Python's `threading` module to achieve this.

   - **Basic Threading Example**:
     ```python
     import PySimpleGUI as sg
     import threading
     import time

     def background_task():
         time.sleep(5)  # Simulate a long-running task
         print("Background task finished!")

     layout = [
         [sg.Text('Task Status:'), sg.Text('', key='-STATUS-')],
         [sg.Button('Start Task')]
     ]
     window = sg.Window('Threading Example', layout)

     while True:
         event, values = window.read()

         if event == sg.WIN_CLOSED:
             break
         elif event == 'Start Task':
             window['-STATUS-'].update('Task started...')
             threading.Thread(target=background_task, daemon=True).start()

     window.close()
     ```

   In this example:
   - When the user clicks "Start Task," a background task (`background_task()`) is started in a separate thread, allowing the GUI to remain responsive.
   - The `daemon=True` parameter ensures that the thread will not block the program from exiting.

2. **Updating the GUI from a Thread**:
   - Direct interaction with the GUI from a background thread is not allowed in most GUI frameworks, including PySimpleGUI, due to thread safety issues. However, PySimpleGUI provides a `window.write_event_value()` method to send messages from a thread to the main event loop.

   - **Updating the GUI Example**:
     ```python
     import PySimpleGUI as sg
     import threading
     import time

     def background_task(window):
         time.sleep(5)  # Simulate a long-running task
         window.write_event_value('-THREAD-', 'Background task finished!')

     layout = [
         [sg.Text('Task Status:')],
         [sg.Text('', key='-STATUS-')],
         [sg.Button('Start Task')]
     ]
     window = sg.Window('Threading with Updates', layout)

     while True:
         event, values = window.read()

         if event == sg.WIN_CLOSED:
             break
         elif event == 'Start Task':
             window['-STATUS-'].update('Task started...')
             threading.Thread(target=background_task, args=(window,), daemon=True).start()
         elif event == '-THREAD-':
             window['-STATUS-'].update(values['-THREAD-'])

     window.close()
     ```

   - In this example:
     - The background task sends an event to the main thread (`window.write_event_value('-THREAD-', message)`) when it completes, allowing the GUI to update.

3. **Async Programming with PySimpleGUI**:
   - Async programming involves using `async` functions with `await` to run tasks asynchronously. PySimpleGUI doesn't natively support `asyncio` directly within the event loop, but you can use `asyncio` in conjunction with threads for background tasks.
   - **Example of Async Programming**:
     ```python
     import PySimpleGUI as sg
     import asyncio

     async def background_task():
         await asyncio.sleep(5)  # Simulate a long-running async task
         return "Background task finished!"

     layout = [
         [sg.Text('Task Status:')],
         [sg.Text('', key='-STATUS-')],
         [sg.Button('Start Task')]
     ]
     window = sg.Window('Async Example', layout)

     while True:
         event, values = window.read(timeout=100)  # Adjust timeout for async events

         if event == sg.WIN_CLOSED:
             break
         elif event == 'Start Task':
             window['-STATUS-'].update('Task started...')
             result = asyncio.run(background_task())  # Run async task
             window['-STATUS-'].update(result)

     window.close()
     ```

   In this example:
   - `asyncio.run(background_task())` is used to run the asynchronous task, but PySimpleGUI's event loop isn't ideal for combining async directly. The `timeout` in `read()` helps periodically check for async task completion.

4. **PySimpleGUI's Async Support**:
   - While PySimpleGUI doesn't have built-in support for async/await tasks, you can integrate `asyncio` with a non-blocking `timeout` to create a responsive GUI.
   - One approach is using `asyncio.create_task()` to schedule background tasks and then updating the GUI when those tasks are complete.

---

### Key Methods and Functions for Async / Threads

1. **`window.write_event_value(event, value)`**:
   - This method is used to send events from a background thread to the main event loop, allowing you to update the GUI.
   - **Example**:
     ```python
     window.write_event_value('-THREAD-', 'Task completed')
     ```

2. **`threading.Thread(target=target_function, daemon=True)`**:
   - Creates a new thread to run `target_function` concurrently with the main thread.
   - **Example**:
     ```python
     threading.Thread(target=background_task, daemon=True).start()
     ```

3. **`asyncio.run(coroutine)`**:
   - Used to run an asynchronous function (coroutine) in a blocking manner, but it may not be optimal for long-running background tasks in PySimpleGUI.
   - **Example**:
     ```python
     result = asyncio.run(background_task())
     ```

4. **`window.read(timeout=100)`**:
   - The `timeout` option in `window.read()` can be used to periodically check the status of tasks, allowing the event loop to continue while waiting for background tasks to complete.
   - **Example**:
     ```python
     event, values = window.read(timeout=100)
     ```

---

### Combining Async and Threads for Better Performance

1. **Handling Long Tasks in Threads**:
   - Long-running tasks can block the GUI if executed in the main thread. By using threads, you can prevent the GUI from freezing while the task runs.
   - Use `window.write_event_value()` to communicate task progress or completion from a thread to the main thread.

2. **Running Background Tasks with Asyncio**:
   - For non-blocking behavior, use `asyncio` to run background tasks in combination with `timeout` in the event loop to periodically check task completion.

---

### Summary

- **Threads** and **Async Programming** in PySimpleGUI allow you to run background tasks concurrently with the main GUI thread, keeping the interface responsive.
- **Threads** are used for executing functions concurrently without blocking the main GUI thread, while **Async Programming** is useful for handling asynchronous tasks like I/O operations or network requests.
- Proper synchronization and communication between threads and the main thread are key, using methods like `window.write_event_value()` to send updates from background tasks to the GUI.