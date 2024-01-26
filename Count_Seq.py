with open('/home/shafayatpiyal/protAlbert/human_mouse.fasta', 'r') as f:
    lines = f.readlines()
    count = 0
    for y in lines:
        if(y[0]=='>'):
            count+=1
print(count)