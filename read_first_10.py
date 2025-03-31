json_file = "../RepControlNet/data/canny_laion/infinity_10k/splits/1.000_000010000.jsonl"
new_path = "../RepControlNet/data/canny_laion/infinity_10k/splits2/1.000_000000010.jsonl"
c = 0
with open(json_file, "r") as f:
    for line in f:
        if c > 10:
            break
        with open(new_path, "a") as f2:
            f2.write(line)
        c += 1
