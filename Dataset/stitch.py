filenames = ['book_{}.txt'.format(i+1) for i in range(7)]
with open('HP_Stitched.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)