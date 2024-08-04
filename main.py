# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# standard library imports
import time
import multiprocessing

# local imports
from batch_processing import batch_process
from fastapi_app import run_server


# ----------------------------------------------------------------------------------------------------------------------
# DEFINE MAIN RUN
# ----------------------------------------------------------------------------------------------------------------------

def main():
    # start FastAPI in separate process
    print("Starting the server... this may take short a while.")
    api_run_process = multiprocessing.Process(target=run_server)
    api_run_process.start()
    time.sleep(5)

    # run batch processing
    print("Starting batch processing now. Please wait.")
    batch_process()
    print("Batch processing complete. Wait for API server to shut down.")

    # stop FastAPI after process is finished
    time.sleep(15)
    api_run_process.terminate()
    api_run_process.join()
    print("The End.")


if __name__ == "__main__":
    main()

