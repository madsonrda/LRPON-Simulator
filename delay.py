import pandas as pd
import sys

filename = sys.argv[1]

delay_df = pd.read_csv("{}".format(filename))
print delay_df['delay'].describe()
