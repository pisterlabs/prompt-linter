import openai

jobname = "ftjob-vdYLsPS7VPcg5HSvpKPdTxm6"
openai.api_key = "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"

r = openai.FineTuningJob.retrieve(jobname)
e = openai.FineTuningJob.list_events(id=jobname, limit=10)
print(f"status: {r}")
print("\n\n\n")
print(f"events: {e}")