
from batch_generator import KerasBatchGenerator


generator = KerasBatchGenerator(4)
g = generator.generate()
for n in g:
    #n = g.next()
    print("Progress:")
    print(generator.current_idx)
    print(generator.folder_index)
    bx = n[0]
    by = n[1]

    sx = bx[0]
    sy = by[0]
    sequence = sx[0]
    direction = sx[1]

    #print(sequence.shape)
    #print(direction)
    #print(sy)
