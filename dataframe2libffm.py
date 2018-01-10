import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

'''
A sci-kit learn inspired script to convert pandas dataframes into libFFM style data.

The script is fairly hacky (hey thats Kaggle) and takes a little while to run a huge dataset.
The key to using this class is setting up the features dtypes correctly for output (ammend transform to suit your needs)

Example below

'''

category_column = ['label','creativeID','userID','positionID','connectionType','telecomsOperator']
numerical_column = ['clickTime']


class FFMFormat:
    def __init__(self):
        self.field_index_ = None
        self.feature_index_ = None
        self.y = None

    def fit(self, df, y=None):
        self.y = y
        df_ffm = df[df.columns.difference([self.y])]
        if self.field_index_ is None:
            self.field_index_ = {col: i for i, col in enumerate(df_ffm)}

        if self.feature_index_ is not None:
            last_idx = max(list(self.feature_index_.values()))

        if self.feature_index_ is None:
            self.feature_index_ = dict()
            last_idx = 0

        for col in df_ffm.columns:
            vals = df_ffm[col].unique()
            for val in vals:
                if pd.isnull(val):
                    continue
                name = '{}_{}'.format(col, val)
                if name not in self.feature_index_:
                    self.feature_index_[name] = last_idx
                    last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row, t):
        ffm = []
        if self.y != None:
            ffm.append(str(row.loc[row.index == self.y][0]))
        if self.y is None:
            ffm.append(str(0))

        for col, val in row.loc[row.index != self.y].to_dict().items():
            col_type = t[col]
            name = '{}_{}'.format(col, val)
            # if col_type.kind == 'O':
            if col in category_column:
                ffm.append('{}:{}:1'.format(self.field_index_[col],
                                            self.feature_index_[name]))
            else:
            # elif col_type.kind == 'i':
                ffm.append('{}:{}:{}'.format(self.field_index_[col],
                                             self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        t = df.dtypes.to_dict()
        return pd.Series(
            {idx: self.transform_row_(row, t) for idx, row in df.iterrows()})


if __name__ == '__main__':
    train_path = 'train.csv'
    train_df = pd.read_csv(train_path, sep=',')
    train_df = train_df.drop(['conversionTime','clickTime'], axis=1)

    ffm_train = FFMFormat()
    ffm_train_data = ffm_train.fit_transform(train_df, y='label')
    ffm_train_data.to_csv(path='train.ffm', index=False)