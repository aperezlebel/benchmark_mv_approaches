
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def ordinal_encode(df, enc=None):

    def encode(df, enc=None):
        if enc is None:
            enc = OrdinalEncoder()
            enc.fit(df)

        data_encoded = enc.transform(df)
        df_encoded = pd.DataFrame(data_encoded,
                                  index=df.index, columns=df.columns)

        return df_encoded, enc

    if isinstance(df, dict):
        return {k: encode(v, enc=enc)[0] for k, v in df.items()}, enc

    if isinstance(df, list):
        return [encode(v, enc=enc)[0] for v in df], enc

    return encode(df, enc=enc)
