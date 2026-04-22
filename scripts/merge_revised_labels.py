#!/usr/bin/env python
"""
Merge revised labels from ChatGPT re-decomposition into metadata_master.csv.

Input:
  - outputs/metadata/metadata_master.csv (current labels, 7870 rows)
  - outputs/non_high_summary_revised_merged_284.json (revised labels for 284 non-high rows)

Output:
  - outputs/metadata/metadata_master.csv (overwritten with merged labels)
  - outputs/metadata/metadata_master_backup_before_merge.csv (backup)

Logic:
  - For each of the 284 revised entries, update:
      raw_label_*, status_*, source_* columns from JSON
      quality_flag from JSON
      exclude_from_main_training = 1 if quality_flag == 'low'
  - 6 low-quality cases (reports redacted) get excluded
  - All other rows remain unchanged
"""

import json
import csv
import codecs
import os
import shutil
from collections import Counter

DISEASES = ['SST', 'IST', 'SSC', 'LHBT', 'IGHL', 'RIPI', 'GHOA']

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_DIR, 'outputs', 'metadata', 'metadata_master.csv')
JSON_PATH = os.path.join(PROJECT_DIR, 'outputs', 'non_high_summary_revised_merged_284.json')
BACKUP_PATH = os.path.join(PROJECT_DIR, 'outputs', 'metadata', 'metadata_master_backup_before_merge.csv')


def main():
    # Load revised labels
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        revised_list = json.load(f)
    revised_map = {entry['exam_id']: entry for entry in revised_list}
    print(f"Loaded {len(revised_list)} revised entries from JSON")

    # Load CSV
    with codecs.open(CSV_PATH, 'r', 'utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    print(f"Loaded {len(rows)} rows from CSV")
    print(f"Columns: {len(fieldnames)}")

    # Backup
    shutil.copy2(CSV_PATH, BACKUP_PATH)
    print(f"Backup saved to {BACKUP_PATH}")

    # Merge
    updated_count = 0
    excluded_count = 0
    change_log = {d: Counter() for d in DISEASES}

    for row in rows:
        eid = row['exam_id']
        if eid not in revised_map:
            continue

        entry = revised_map[eid]
        updated_count += 1

        # Update quality_flag
        old_qf = row['quality_flag']
        new_qf = entry.get('quality_flag', old_qf)
        row['quality_flag'] = new_qf

        # Exclude low-quality cases
        if new_qf == 'low':
            row['exclude_from_main_training'] = '1'
            excluded_count += 1

        # Update labels, status, source per disease
        for d in DISEASES:
            label_col = f'raw_label_{d}'
            status_col = f'status_{d}'
            source_col = f'source_{d}'

            old_label = row.get(label_col, '')
            new_label = str(entry['labels'].get(d, old_label))
            if old_label != new_label:
                change_log[d][f'{old_label}->{new_label}'] += 1
            row[label_col] = new_label

            # Update status
            new_status = entry.get('label_status', {}).get(d)
            if new_status is not None:
                row[status_col] = new_status

            # Update source
            new_source = entry.get('source_summary', {}).get(d)
            if new_source is not None:
                row[source_col] = new_source

    # Write back
    with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Report
    print(f"\n=== Merge Complete ===")
    print(f"Updated rows: {updated_count}")
    print(f"Excluded (low quality): {excluded_count}")
    print()

    total_changes = 0
    print("Label changes per disease:")
    for d in DISEASES:
        changes = change_log[d]
        n = sum(changes.values())
        total_changes += n
        if n > 0:
            print(f"  {d}: {n} changes - {dict(changes)}")
        else:
            print(f"  {d}: no changes")
    print(f"\nTotal label changes: {total_changes}")

    # Verify
    print("\n=== Post-merge verification ===")
    with codecs.open(CSV_PATH, 'r', 'utf-8-sig') as f:
        reader = csv.DictReader(f)
        verify_rows = list(reader)
    print(f"Output rows: {len(verify_rows)}")

    qf_counts = Counter(r['quality_flag'] for r in verify_rows)
    print(f"Quality flags: {dict(qf_counts)}")

    excl_count = sum(1 for r in verify_rows if r.get('exclude_from_main_training') == '1')
    print(f"Excluded from training: {excl_count}")

    for d in DISEASES:
        col = f'raw_label_{d}'
        vc = Counter(r[col] for r in verify_rows)
        print(f"  {d}: {dict(vc)}")


if __name__ == '__main__':
    main()
