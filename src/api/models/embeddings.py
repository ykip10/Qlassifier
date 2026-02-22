import time

from InstructorEmbedding import INSTRUCTOR

print("Loading INSTRUCTOR model...")
start_time = time.perf_counter()
instructor = INSTRUCTOR("hkunlp/instructor-large")
end_time = time.perf_counter()
print(f"Finished loading ({end_time - start_time:.2f} secs)")

