ADMIT = 'admit'
GRE   = 'gre'
GPA   = 'gpa'
RANK  = 'rank'

POSSIBLE_VALUES = {
    RANK: 4,
    GRE: 2,
    GPA: 2,
    ADMIT: 2
}

class BayesNetwork():

    def _table_rank(self, df):
        return self._table(df, [], RANK)

    def _table_gre(self, df):
        return self._table(df, [RANK], GRE)

    def _table_gpa(self, df):
        return self._table(df, [RANK], GPA)

    def _table_admit(self, df):
        return self._table(df, [RANK, GRE, GPA], ADMIT)

    ## ["rank", "GRE", "GPA"],  "admit", {"rank":4, "GRE":2, "GRE":2, "GRE":2}
    def _table(self, df, fixed_columns, table_name):
        counts = {}
        table = {}

        for idx, row in df.iterrows():
            ## first row: rank=0, GRE=1, GPA=1, admit=0
            fixed_columns_values = []
            if fixed_columns:
                for column_name in fixed_columns:
                    column_value = row[column_name]
                    fixed_columns_values.append(column_value)
                pre_k = tuple(fixed_columns_values)
            else:
                fixed_columns_values.append(0)  ## dummy value for rank case
                pre_k = tuple(fixed_columns_values)
                

            k = fixed_columns_values.append(row[table_name]) ## [0, 1, 1, 0]
            k = tuple(fixed_columns_values)

            # P(admit=0 | rank=0, GRE=1, GPA=1)
            
            if pre_k not in counts:
                counts[pre_k] = 0
            counts[pre_k] += 1

            if k not in table:
                table[k] = 0
            table[k] += 1

        for k, c in table.items():
            # if(len(list(k)[:-1]) == 1):
            #     pre_k = list(k)[:-1][0]
            # else:
            pre_k = tuple(list(k)[:-1])
            table[k] = (c + 1) / (counts[pre_k] + POSSIBLE_VALUES[table_name])
            # Aplica laplace

        return counts, table

    def train(self, df):
        tables = {}
        tables[RANK]  = self._table_rank(df)
        tables[GRE]   = self._table_gre(df)
        tables[GPA]   = self._table_gpa(df)
        tables[ADMIT] = self._table_admit(df)
        self.tables = tables

    def _table_value(self, name, k):
        c, t = self.tables[name]
        if name == RANK:
            k = [0, k[0]]
        pre_k = tuple(k[:-1])
        k = tuple(k)
        return t.get(k, 1 / (c.get(pre_k, 0) + POSSIBLE_VALUES[name]))
        
    # p(A = 1 |R=2,GRE=1,GPA=0) = 
    #              df, {RANK: 2, GRE: 1, GPA:0}, {ADMIT:1} 
    def eval(self, df, in_                     , out_):
        tables = self.tables
        
        for name, value in in_.items():
            table = self._get_table(name)

    def _conjunta(self, rank, gre, gpa, admit):
        return self._table_value(RANK, [rank]) * self._table_value(GPA, [rank, gpa]) * self._table_value(GRE, [rank, gre]) * self._table_value(ADMIT, [rank, gre, gpa, admit])

    ## a
    # P(A=0 | R=1) = P(A=0 y R=1) / P(R=1) = GRE∈{0,1} GPA∈{0,1} P(A=0, GRE, GPA, R = 1)
    def a(self):
        rank = 1
        admit = 0
        up = 0
        down = self._table_value(RANK, [rank])
        for gre in [0, 1]:
            for gpa in [0, 1]:
                up += self._conjunta(rank, gre, gpa, admit)
        return up / down

    ## b
    # Calcular la probabilidad de que una persona que fue a una escuela de rango 2, tenga
    # GRE = 450 y GPA = 3.5 sea admitida en la universidad.
    # P (A = 1 | R=2, GRE=0, GPA=1) = P(A=1, R=2, GRE=0, GPA=1) / P(R=2, GRE=0, GPA=1)
    def b(self):
        rank = 2
        admit = 1
        gre = 0
        gpa = 1
        up = self._conjunta(rank, gre, gpa, admit)
        down = 0
        for _admit in [0, 1]:
            down += self._conjunta(rank, gre, gpa, _admit)
        return up / down

    def all(self):
        s = 0
        for rank in [1, 2, 3, 4]:
            for admit in [0, 1]:
                for gre in [0, 1]:
                    for gpa in [0, 1]:
                        s += self._conjunta(rank, gre, gpa, admit)
        return s
