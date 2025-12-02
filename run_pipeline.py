import os
from pipeline import run_pipeline  # Ensure 'run_pipeline' exists in your pipeline.py
# If you don't have a main function in pipeline.py, copy your update logic here.

if __name__ == "__main__":
    print("🚀 Starting Scheduled Data Update...")
    
    try:
        # This calls your main logic to fetch GDELT and upload to Snowflake
        run_pipeline() 
        print("✅ Update Success")
    except Exception as e:
        print(f"❌ Update Failed: {e}")
        exit(1)