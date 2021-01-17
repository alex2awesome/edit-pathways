#!/usr/bin/env python3

"""
Ugly little program to turn a diffengine database into a csv. Maybe something
prettier should should be part of diffengine itself?
"""

import os
import re
import csv
import glob
import sqlite3

from os.path import join, basename

def dump(db_file):
    csv_file = join('csv', basename(db_file).replace('.db', '.csv'))
    print('{} -> {}'.format(db_file, csv_file))
    
    db = sqlite3.connect(db_file)
    out = csv.writer(open(csv_file, 'w'))
    out.writerow(['url', 'old', 'new'])

    for old_id, new_id in db.execute('SELECT old_id, new_id FROM diff'):

        url, old_url = db.execute('SELECT url, archive_url FROM entryversion WHERE id = ?', [old_id]).fetchone()
        new_url = db.execute('SELECT archive_url FROM entryversion WHERE id = ?', [new_id]).fetchone()[0]

        if not (new_url and old_url):
            continue

        out.writerow([url, old_url, new_url])

for db_file in glob.glob('db/*.db'):
    dump(db_file)
