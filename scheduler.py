import streamlit as st
import time
import joblib
from apscheduler.schedulers.background import BackgroundScheduler
from queue import Queue
import numpy as np

# Load the AI magic
ml_model = joblib.load('conversation_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Fixed tasks: (task, duration in seconds)
tasks = [
    ("Reading", 5),
    ("Eating", 3),
    ("Homework", 6)
]

# Queue for log chatter
log_queue = Queue()

def log(message):
    """Toss a message in the log pile with a timestamp."""
    log_queue.put(f"{time.strftime('%H:%M:%S')}: {message}")

def execute_task(task, duration, accept=True, progress_bar=None):
    """Kick off a task or let the AI roast you if you ditch it."""
    if accept:
        log(f"{task} time! Running for {duration} seconds...")
        for i in range(duration):
            time.sleep(1)  # Chill for a sec
            if progress_bar:
                progress_bar.progress((i + 1) / duration)  # Groove that bar
        log(f"{task} done—nailed it!")
        if progress_bar:
            progress_bar.progress(1.0)  # Full send!
        st.session_state.rejection_count = 0  # Reset counter on accept
    else:
        log(f"You rejected {task}? Alright:-")
        task_vector = tfidf_vectorizer.transform([task])
        response = ml_model.predict(task_vector)[0]
        log(f"AI says: {response}")
        st.session_state.rejection_count += 1  # Bump rejection counter
        if st.session_state.rejection_count >= 3:
            st.warning("Hey, 3 rejections in a row? You get a warning!")

def schedule_tasks(scheduler, task_list, progress_dict):
    """Line up tasks to roll on their own."""
    for task, duration in task_list:
        scheduler.add_job(execute_task, 'interval', seconds=duration + 5,  # Buffer for breathing room
                          args=(task, duration, True, progress_dict[task]), id=task)

# Streamlit UI—let’s get human
st.title("Task Scheduler")

# Sidebar with quirky instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
Hey there, Sukrati Ma’am! 

1. **Peek Below**: "Reading", "Eating", "Homework" are ready to roll—check the lineup!
2. **Manual Jam**: Pick a task from "Select Task", tick "Accept Task" to do it, or ditch it (uncheck) to get AI response back.
3. **Auto Mode**: Smack "Start Scheduler" to let these bad boys run on repeat. "Stop Scheduler" kills the vibe.
4. **Rejection Alert**: Ditch three tasks in a row manually? You’ll get a shout-out—watch the screen!

""")

# Scheduler controls
st.sidebar.header("Scheduler Controls")
start_scheduler = st.sidebar.button("Start Scheduler")
stop_scheduler = st.sidebar.button("Stop Scheduler")

# Manual task execution
st.sidebar.header("Manual Execution")
selected_task = st.sidebar.selectbox("Select Task", ["None"] + [t[0] for t in tasks])
accept_task = st.sidebar.checkbox("Accept Task", value=True)
run_task = st.sidebar.button("Run Task Now")

# Session state—keeping it real
if 'scheduler' not in st.session_state:
    st.session_state.scheduler = BackgroundScheduler()
    st.session_state.running = False
if 'progress_bars' not in st.session_state:
    st.session_state.progress_bars = {}
if 'logs' not in st.session_state:
    st.session_state.logs = []  # Log stash
if 'rejection_count' not in st.session_state:
    st.session_state.rejection_count = 0  # Track consecutive rejections

# Tasks and progress—front and center
st.header("Tasks and Progress")
st.table(tasks)  # Rock-solid table

# Progress bars—fancy moves
progress_container = st.empty()
def update_progress_bars():
    """Set those bars in motion."""
    progress_bars = {}
    for task, duration in tasks:
        if task not in st.session_state.progress_bars:
            st.session_state.progress_bars[task] = progress_container.progress(0)
        progress_bars[task] = st.session_state.progress_bars[task]
    st.session_state.progress_bars = progress_bars

# Log display—chatty corner
st.header("Logs-")
log_container = st.empty()

def update_logs():
    """Stack logs and spill the tea."""
    while not log_queue.empty():
        st.session_state.logs.append(log_queue.get())
    log_container.text("\n".join(st.session_state.logs))  # Dump it all

# Start the scheduler—go time
if start_scheduler and not st.session_state.running:
    update_progress_bars()
    schedule_tasks(st.session_state.scheduler, tasks, st.session_state.progress_bars)
    st.session_state.scheduler.start()
    st.session_state.running = True
    log("Scheduler’s alive—tasks incoming!")

# Stop the scheduler—chill pill
if stop_scheduler and st.session_state.running:
    st.session_state.scheduler.shutdown()
    st.session_state.running = False
    log("Scheduler’s out—peace!")
    for bar in st.session_state.progress_bars.values():
        bar.progress(0)  # Reset the groove

# Manual run—your call
if run_task and selected_task != "None":
    update_progress_bars()
    task_info = next(t for t in tasks if t[0] == selected_task)
    task, duration = task_info
    progress_bar = st.session_state.progress_bars[task]
    execute_task(task, duration, accept_task, progress_bar)
    update_logs()

# Keep it rolling
if st.session_state.running:
    while st.session_state.running:
        update_logs()
        time.sleep(1)
else:
    update_progress_bars()
    update_logs()

if __name__ == "__main__":
    st.write("Ready to roll, Less Goo!")