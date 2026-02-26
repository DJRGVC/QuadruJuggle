from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys

path = sys.argv[1]
ea = EventAccumulator(path)
ea.Reload()
for tag in ea.Tags()['scalars']:
    rows = ea.Scalars(tag)
    print(f'=== {tag} ===')
    for r in rows:
        print(f'{r.step},{r.value}')
