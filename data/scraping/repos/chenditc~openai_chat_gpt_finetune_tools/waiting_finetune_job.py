import fire
import openai
import time

def waiting_finetune_job(job_id=None):
    if job_id is None:
        print("No job id specified, waiting for the first running job")
        finetune_job_list = openai.FineTuningJob.list(limit=10)
        for job in finetune_job_list.data:
            print(job)
            if job.status == "running":
                job_id = job.id
                break

    if job_id is None:
        print("No job to monitor")
        exit()
    
    status = "running"
    while status == "running":
        job_info = openai.FineTuningJob.retrieve(job_id)
        train_time = int(time.time() - job_info.created_at)
        print(f"Trained {train_time} seconds\n {job_info}")
        status = job_info.status
        time.sleep(10)

    total_training_time = job_info.finished_at - job_info.created_at
    print(f"Total training time: {total_training_time} seconds")

if __name__ == "__main__":
    fire.Fire(waiting_finetune_job)