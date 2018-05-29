import numpy as np

class CsvWriter:
    def __init__(self, nline):
        self.nline = nline
        self.header_set = list()
        self.content_set = list()
        for i in range(self.nline):
            self.content_set.append(list())

    def add_header(self, header):
        self.header_set.append(header)
    
    def add_content(self, index, content):
        self.content_set[index].append(content)

    def write(self, path):
        print("Write csv files on %s"%path)
        with open(path, 'w') as f:
            f.write(','.join([str(v) for v in self.header_set])+'\n')
            for i in range(self.nline):
                f.write(','.join([str(v) for v in self.content_set[i]])+'\n')

class CsvWriter2:
    def __init__(self, nline):
        self.nline = nline
        self.header_set = list()
        self.content_set = list()
        for i in range(self.nline):
            self.header_set.append(list())
            self.content_set.append(list())

    def add_header(self, index, header):
        self.header_set[index].append(header)
    
    def add_content(self, index, content):
        self.content_set[index].append(content)

    def write(self, path):
        print("Write csv files on %s"%path)
        
        max_length = 0
        for i in range(self.nline):
            max_length = max(max_length, max([len(str(v)) for v in self.header_set[i]]))
            max_length = max(max_length, max([len(str(v)) for v in self.content_set[i]]))

        with open(path, 'w') as f:
            for i in range(self.nline):
                f.write(','.join([str(v).rjust(max_length, ' ') for v in self.header_set[i]])+'\n')
                f.write(','.join([str(v).rjust(max_length, ' ') for v in self.content_set[i]])+'\n')

class CsvWriter3:
    def __init__(self):
        self.content_dict = dict()

    def add_content(self, row_key, column_key, content):
        if row_key not in self.content_dict.keys():
            self.content_dict[row_key] = dict()
        self.content_dict[row_key][column_key] = content

    def write(self, path, row_key_set, col_key_set):
        print("Write csv files on %s"%path)
        
        nrow = len(row_key_set)

        col_header = list()
        col_header.append('')
        for col_key in col_key_set:
            col_header.append(str(col_key))

        content_list = list()
        for row_key in row_key_set:
            tmp = list()
            tmp.append(str(row_key))
            for col_key in col_key_set:
                tmp.append(self.content_dict[row_key][col_key])
            content_list.append(tmp) 

        max_len_list = list()
        nrow, ncol = len(content_list), len(content_list[0])

        for c_idx in range(ncol):
            tmp = len(col_header[c_idx])
            for r_idx in range(nrow):
                tmp = max(tmp, len(str(content_list[r_idx][c_idx])))
            max_len_list.append(tmp) # max_len_list[c_idx] = tmp

        with open(path, 'w') as f:
            f.write(','.join([str(col_header[c_idx]).rjust(max_len_list[c_idx], ' ') for c_idx in range(ncol)])+'\n')
            for r_idx in range(nrow):
                f.write(','.join([str(content_list[r_idx][c_idx]).rjust(max_len_list[c_idx], ' ') for c_idx in range(ncol)])+'\n')

def read_csv(path):
    '''
    Complementary to CsvWriter
    '''
    content_set = list()
    with open(path, 'r') as lines:
        reads = list()
        for line in lines: 
            reads.append(line)
        header_set = reads[0][:-1]
        for read in reads[1:]:
            content_set.append(read[:-1].split(','))
    return header_set, content_set
