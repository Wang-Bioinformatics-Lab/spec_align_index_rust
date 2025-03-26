import numba
import numpy as np
from numba.typed import List
import os
import concurrent.futures
import pymzml
import time
import argparse

# Constants
TOLERANCE = 0.01
SHIFTED_OFFSET = 6000
TOPK = 10
MINMATCHES = 6
THRESHOLD = 0.7
TOPPRODUCTS = 30000
ADJACENT_BINS = np.array([-1, 0, 1], dtype=np.int64)

SpectrumTuple = numba.types.Tuple([
    numba.float32[:],  # mz
    numba.float32[:],  # intensity
    numba.float32,     # precursor_mz
    numba.int32        # precursor_charge
])


def filter_peaks_optimized(mz_array, intensity_array, precursor_mz, precursor_charge):
    """
    Keep a peak i if:
      - It is among top 6 by intensity within ±25 Da of mz[i].
      - Also |mz[i] - precursor_mz| > 17.
    Then sqrt-transform intensities & L2-normalize.
    """
    N = len(mz_array)
    if N == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            precursor_mz,
            precursor_charge
        )

    combo = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        combo[i, 0] = mz_array[i]
        combo[i, 1] = intensity_array[i]
        combo[i, 2] = i
    combo = combo[np.argsort(combo[:,0])]

    kept_idx = set()
    left = 0
    right= 0
    for i in range(N):
        mz_i = combo[i, 0]
        while left < N and combo[left, 0] < (mz_i - 25):
            left += 1
        while right < N and combo[right, 0] <= (mz_i + 25):
            right += 1
        subrange = combo[left:right]
        if len(subrange) <= 6:
            top_idx = [int(x[2]) for x in subrange]
        else:
            sorted_sub = subrange[subrange[:,1].argsort()[::-1]]
            top_idx = [int(x[2]) for x in sorted_sub[:6]]
        if int(combo[i,2]) in top_idx:
            if abs(mz_i - precursor_mz)>17:
                kept_idx.add(int(combo[i,2]))

    final_mz = []
    final_int= []
    for i in range(N):
        if i in kept_idx:
            final_mz.append(mz_array[i])
            final_int.append(intensity_array[i])
    final_mz = np.array(final_mz, dtype=np.float32)
    final_int= np.array(final_int, dtype=np.float32)

    final_int= np.sqrt(final_int)
    norm = np.linalg.norm(final_int)
    if norm !=0:
        final_int= final_int/norm
    # norm to 1
    # max_val = np.max(final_int) if final_int.size > 0 else 0
    # if max_val != 0:
    #     final_int = final_int / max_val
    sorted_indices = np.argsort(final_mz)
    return (
        np.array(final_mz)[sorted_indices].astype(np.float32),
        np.array(final_int)[sorted_indices].astype(np.float32),
        precursor_mz,
        precursor_charge
    )

###############################################################################
# 2) File Parsing (MGF / mzML)
###############################################################################

def parse_mgf_file(path):
    results=[]
    with open(path,'r') as f:
        in_ions=False
        current={"mz":[],"int":[],"pepmass":None,"charge":None}
        for line in f:
            line=line.strip()
            if not line:
                continue
            if line=="BEGIN IONS":
                in_ions=True
                current={"mz":[],"int":[],"pepmass":None,"charge":None}
            elif line=="END IONS":
                if current["pepmass"] and current["charge"]:
                    pepmass_val=float(current["pepmass"])
                    cstr = current["charge"].rstrip("+")
                    if cstr.isdigit():
                        charge_val=int(cstr)
                    else:
                        charge_val=1
                    if len(current["mz"])>0:
                        mz_arr=np.array(current["mz"],dtype=np.float32)
                        in_arr=np.array(current["int"],dtype=np.float32)
                        filtered=filter_peaks_optimized(mz_arr,in_arr,pepmass_val,charge_val)
                        if len(filtered[0])>0:
                            results.append(filtered)
                in_ions=False
            else:
                if in_ions:
                    if "=" in line:
                        key,val=line.split("=",1)
                        key=key.lower()
                        if key=="pepmass":
                            val_parts=val.split()
                            current["pepmass"]=val_parts[0]
                        elif key=="charge":
                            current["charge"]=val
                    else:
                        parts=line.split()
                        if len(parts)==2:
                            try:
                                mz_v=float(parts[0])
                                in_v=float(parts[1])
                                current["mz"].append(mz_v)
                                current["int"].append(in_v)
                            except:
                                pass
    return results

def parse_mzml_file(path):
    run = pymzml.run.Reader(path)
    results=[]
    for spec in run:
        if spec.ms_level==2:
            if not spec.selected_precursors:
                continue
            mz_arr=np.array(spec.mz,dtype=np.float32)
            in_arr=np.array(spec.i,dtype=np.float32)
            precursor_mz=float(spec.selected_precursors[0]['mz'])
            precursor_charge=spec.selected_precursors[0].get('charge',1)
            if not isinstance(precursor_charge,int):
                try:
                    precursor_charge=int(precursor_charge)
                except:
                    precursor_charge=1
            filtered=filter_peaks_optimized(mz_arr,in_arr,precursor_mz,precursor_charge)
            if len(filtered[0])>0:
                results.append(filtered)
    return results

def parse_one_file(path):
    ext=os.path.splitext(path)[1].lower()
    if ext==".mgf":
        return parse_mgf_file(path)
    elif ext==".mzml":
        return parse_mzml_file(path)
    else:
        # fallback
        try:
            return parse_mzml_file(path)
        except:
            return parse_mgf_file(path)

def parse_files_in_parallel(file_paths, threads=1):
    if threads<=1 or len(file_paths)<=1:
        all_spectra=[]
        for fp in file_paths:
            parsed=parse_one_file(fp)
            all_spectra.extend(parsed)
        return all_spectra
    else:
        all_spectra=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as ex:
            futs=[ex.submit(parse_one_file,fp) for fp in file_paths]
            for fut in concurrent.futures.as_completed(futs):
                all_spectra.extend(fut.result())
        return all_spectra


@numba.njit
def create_index(spectra, is_shifted):
    """Create index structure for fast lookups"""
    entries = List()
    for spec_idx in range(len(spectra)):
        spec = spectra[spec_idx]
        mz_arr = spec[0]
        int_arr = spec[1]
        precursor_mz = spec[2]

        for peak_idx in range(len(mz_arr)):
            mz = mz_arr[peak_idx]
            intensity = int_arr[peak_idx]

            if is_shifted:
                bin_val = np.int64(round(
                    (precursor_mz - mz + SHIFTED_OFFSET) / TOLERANCE
                ))
            else:
                bin_val = np.int64(round(mz / TOLERANCE))

            entries.append((bin_val, spec_idx, peak_idx, mz, intensity))

    # Sort by bin and spec_idx
    entries.sort()
    return entries

@numba.njit
def find_bin_range(entries, target_bin):
    """Binary search for bin ranges"""
    left = 0
    right = len(entries)
    while left < right:
        mid = (left + right) // 2
        if entries[mid][0] < target_bin:
            left = mid + 1
        else:
            right = mid
    start = left

    right = len(entries)
    while left < right:
        mid = (left + right) // 2
        if entries[mid][0] <= target_bin:
            left = mid + 1
        else:
            right = mid
    return start, left

@numba.njit
def compute_all_pairs(spectra, shared_entries, shifted_entries):
    """Numba-accelerated all-vs-all comparison"""
    results = List()
    n_spectra = len(spectra)

    for query_idx in range(n_spectra):
        query_spec = spectra[query_idx]
        upper_bounds = np.zeros(n_spectra, dtype=np.float32)
        match_counts = np.zeros(n_spectra, dtype=np.int32)

        # Process both shared and shifted peaks
        for peak_idx in range(len(query_spec[0])):
            mz = query_spec[0][peak_idx]
            intensity = query_spec[1][peak_idx]
            precursor_mz = query_spec[2]

            # Shared peaks processing
            shared_bin = np.int64(round(mz / TOLERANCE))
            shifted_bin = np.int64(round((precursor_mz - mz + SHIFTED_OFFSET) / TOLERANCE))

            # Check both shared and shifted entries
            for entries, bin_val in [(shared_entries, shared_bin),
                                     (shifted_entries, shifted_bin)]:
                for delta in ADJACENT_BINS:
                    target_bin = bin_val + delta
                    start, end = find_bin_range(entries, target_bin)
                    pos = start
                    while pos < end and entries[pos][1] <= query_idx:
                        pos += 1
                    # Find matches in this bin
                    while pos < end and entries[pos][0] == target_bin:
                        spec_idx = entries[pos][1]
                        upper_bounds[spec_idx] += intensity * entries[pos][4]
                        match_counts[spec_idx] += 1
                        pos += 1

        # Collect candidates
        candidates = List()
        for spec_idx in range(query_idx + 1, n_spectra):
            if (upper_bounds[spec_idx] >= THRESHOLD and match_counts[spec_idx] >= MINMATCHES):
                candidates.append((spec_idx, upper_bounds[spec_idx]))

        # Sort candidates by upper bound score
        candidates.sort(key=lambda x: -x[1])

        # Process top candidates for exact matching
        exact_matches = List()
        for spec_idx, _ in candidates[:TOPPRODUCTS * 2]:
            target_spec = spectra[spec_idx]
            score, shared, shifted = calculate_exact_score(spectra[query_idx], target_spec)
            if score >= THRESHOLD:
                exact_matches.append((spec_idx, score, shared, shifted))

        # Sort and store top results
        exact_matches.sort(key=lambda x: -x[1])
        results.append((query_idx, exact_matches[:TOPPRODUCTS]))

    return results


@numba.njit
def calculate_exact_score_GNPS(query_spec, target_spec):
    """Numba-optimized cosine scoring with shift handling"""
    q_mz = query_spec[0]
    q_int = query_spec[1]
    q_pre = query_spec[2]

    t_mz = target_spec[0]
    t_int = target_spec[1]
    t_pre = target_spec[2]

    # Calculate precursor mass difference (assuming charge=1)
    precursor_mass_diff = q_pre - t_pre
    allow_shift = True
    fragment_tol = TOLERANCE

    # Pre-allocate arrays for matches (adjust size as needed)
    max_matches = len(q_mz) * 2  # Estimate maximum possible matches
    scores_arr = np.zeros(max_matches, dtype=np.float32)
    idx_q = np.zeros(max_matches, dtype=np.int32)
    idx_t = np.zeros(max_matches, dtype=np.int32)
    match_count = 0

    # For each peak in query spectrum
    for q_idx in range(len(q_mz)):
        q_mz_val = q_mz[q_idx]
        q_int_val = q_int[q_idx]

        # For each possible shift (charge=1)
        num_shifts = 1
        if allow_shift and abs(precursor_mass_diff) >= fragment_tol:
            num_shifts += 1

        for shift_idx in range(num_shifts):
            if shift_idx == 0:
                # No shift
                adjusted_mz = q_mz_val
            else:
                # Apply shift
                adjusted_mz = q_mz_val - precursor_mass_diff

            # Find matching peaks in target using binary search
            start = np.searchsorted(t_mz, adjusted_mz - fragment_tol)
            end = np.searchsorted(t_mz, adjusted_mz + fragment_tol)

            for t_idx in range(start, end):
                if match_count >= max_matches:
                    break  # Prevent overflow

                # Store match information
                scores_arr[match_count] = q_int_val * t_int[t_idx]
                idx_q[match_count] = q_idx
                idx_t[match_count] = t_idx
                match_count += 1

    # Sort matches by score descending using argsort
    if match_count == 0:
        return 0.0, 0.0, 0.0

    # Get valid matches
    valid_scores = scores_arr[:match_count]
    valid_q_idx = idx_q[:match_count]
    valid_t_idx = idx_t[:match_count]

    # Argsort descending
    sort_order = np.argsort(-valid_scores)

    # Track used peaks
    q_used = np.zeros(len(q_mz), dtype=np.bool_)
    t_used = np.zeros(len(t_mz), dtype=np.bool_)
    total = 0.0
    shared = 0.0
    shifted = 0.0

    # Accumulate top matches
    for i in sort_order:
        q_idx = valid_q_idx[i]
        t_idx = valid_t_idx[i]

        if not q_used[q_idx] and not t_used[t_idx]:
            score = valid_scores[i]
            total += score

            # Determine match type
            mz_diff = abs(q_mz[q_idx] - t_mz[t_idx])
            if mz_diff <= fragment_tol:
                shared += score
            else:
                shifted += score

            q_used[q_idx] = True
            t_used[t_idx] = True

    return total, shared, shifted


@numba.njit
def calculate_exact_score(query_spec, target_spec):
    """Calculate exact cosine similarity with tolerance checks"""
    # Access tuple elements by index
    q_mz = query_spec[0]  # Already sorted by mz
    q_int = query_spec[1]
    q_pre = query_spec[2]

    t_mz = target_spec[0]  # Already sorted by mz
    t_int = target_spec[1]
    t_pre = target_spec[2]

    q_used = np.zeros(len(q_mz), dtype=np.bool_)
    t_used = np.zeros(len(t_mz), dtype=np.bool_)
    total = 0.0
    shared = 0.0
    shifted = 0.0

    # 1. Shared peak matching
    for q_idx in range(len(q_mz)):
        if q_used[q_idx]:
            continue

        mz_q = q_mz[q_idx]
        # Find matches using binary search on sorted mz
        start = np.searchsorted(t_mz, mz_q - TOLERANCE)
        end = np.searchsorted(t_mz, mz_q + TOLERANCE)

        best_score = 0.0
        best_t_idx = -1
        for t_idx in range(start, end):
            if not t_used[t_idx]:
                current_score = q_int[q_idx] * t_int[t_idx]
                if current_score > best_score:
                    best_score = current_score
                    best_t_idx = t_idx

        if best_t_idx != -1:
            total += best_score
            shared += best_score
            q_used[q_idx] = True
            t_used[best_t_idx] = True

    # 2. Shifted peak matching (critical fix)
    # Create and sort shifted mz arrays while tracking original indices
    # ---------------------------------------------------------------
    # For query
    q_shifted = q_pre - q_mz + SHIFTED_OFFSET
    q_shifted_sorted_idx = np.argsort(q_shifted)
    q_shifted_sorted = q_shifted[q_shifted_sorted_idx]

    # For target
    t_shifted = t_pre - t_mz + SHIFTED_OFFSET
    t_shifted_sorted_idx = np.argsort(t_shifted)
    t_shifted_sorted = t_shifted[t_shifted_sorted_idx]
    # ---------------------------------------------------------------

    # Match using sorted shifted mz arrays
    for q_pos in range(len(q_shifted_sorted)):
        q_orig_idx = q_shifted_sorted_idx[q_pos]
        if q_used[q_orig_idx]:
            continue

        mz_q = q_shifted_sorted[q_pos]
        start = np.searchsorted(t_shifted_sorted, mz_q - TOLERANCE)
        end = np.searchsorted(t_shifted_sorted, mz_q + TOLERANCE)

        best_score = 0.0
        best_t_pos = -1
        for t_pos in range(start, end):
            t_orig_idx = t_shifted_sorted_idx[t_pos]
            if not t_used[t_orig_idx]:
                current_score = q_int[q_orig_idx] * t_int[t_orig_idx]
                if current_score > best_score:
                    best_score = current_score
                    best_t_pos = t_pos

        if best_t_pos != -1:
            t_orig_idx = t_shifted_sorted_idx[best_t_pos]
            total += best_score
            shifted += best_score
            q_used[q_orig_idx] = True
            t_used[t_orig_idx] = True

    return total, shared, shifted

# The preprocessing functions you provided would go here
# [Include your filter_peaks_optimized, parse_mgf_file, etc. here]
def parse_arguments():
    parser= argparse.ArgumentParser(
        description="Flattened all-pairs with nested dictionary indexing, shift always on."
    )
    parser.add_argument("-t","--input_files",nargs="+",required=True,help="MGF/mzML file paths")
    parser.add_argument("--threads",type=int,default=1,help="Number of parallel threads in Numba + parse")
    parser.add_argument("-o","--output_file",default="output.tsv",help="Output file")
    return parser.parse_args()
def main():
    args= parse_arguments()
    # Parse input files using your existing code
    if args.threads>0:
        threads =args.threads
    spectra_list = parse_files_in_parallel(args.input_files)
    print("complete parse all the spectra")

    # Convert to Numba-compatible format
    numba_spectra = List()
    for spec in spectra_list:
        numba_spectra.append((
            spec[0].astype(np.float32),
            spec[1].astype(np.float32),
            np.float32(spec[2]),
            np.int32(spec[3])
        ))
    # Build indexes
    print("Building indexes...")
    shared_idx = create_index(numba_spectra, False)
    shifted_idx = create_index(numba_spectra, True)
    start_time = time.time()
    # Compute matches
    print("Computing matches...")
    matches = compute_all_pairs(numba_spectra, shared_idx, shifted_idx)
    end_time = time.time()
    print("All pairwise compute time:", end_time - start_time)

    # Write results
    print("Writing output...")
    with open(args.output_file, 'w') as f:
        f.write("scan1\tmz1\tscan2\tmz2\tscore\tshared\tshifted\n")
        for query_idx, candidates in matches:
            q_spec = numba_spectra[query_idx]
            for cand in candidates:
                t_spec = numba_spectra[cand[0]]
                f.write(f"{query_idx}\t{q_spec[2]:.4f}\t"  # Index 2: precursor_mz
                        f"{cand[0]}\t{t_spec[2]:.4f}\t"  # Index 2: precursor_mz
                        f"{cand[1]:.4f}\t{cand[2]:.4f}\t{cand[3]:.4f}\n")

    print("Done!")


if __name__ == "__main__":
    main()