import argparse

def read_numbers(file_path):
    with open(file_path, 'r') as f:
        return [float(line.strip()) for line in f]

def get_top_n_pairs(file1, file2, file3, file4, n):
    nums1 = read_numbers(file1)
    nums2 = read_numbers(file2)
    mods1 = read_numbers(file3)
    mods2 = read_numbers(file4)

    if not (len(nums1) == len(nums2) == len(mods1) == len(mods2)):
        raise ValueError("All files must have the same number of lines.")

    # Build a list of tuples: ((a, b), (c, d))
    all_data = [((a, b), (c, d)) for a, b, c, d in zip(nums1, nums2, mods1, mods2)]

    # Sort based on sum of a + b
    top_n = sorted(all_data, key=lambda x: x[0][0] + x[0][1], reverse=False)[:n]
    return top_n

def main():
    parser = argparse.ArgumentParser(description='Get top N number pairs with modification values.')
    parser.add_argument('file_1', type=str, help='Path to first input file (a values)')
    parser.add_argument('file_2', type=str, help='Path to second input file (b values)')
    parser.add_argument('file_3', type=str, help='Path to third input file (c modifiers)')
    parser.add_argument('file_4', type=str, help='Path to fourth input file (d modifiers)')
    parser.add_argument('n', type=int, help='Number of top pairs to retrieve')

    args = parser.parse_args()

    top_n_pairs = get_top_n_pairs(args.file_1, args.file_2, args.file_3, args.file_4, args.n)

    print("Top pairs with modifications (a ± c, b ± d):")
    for (a, b), (c, d) in top_n_pairs:
        print(f"({a} ± {c}, {b} ± {d})")

if __name__ == '__main__':
    main()

