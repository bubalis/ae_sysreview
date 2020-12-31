# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:06:47 2020

@author: benja
"""




import find_cites
import authors_326
import assign_topic_cols
import pandas as pd
#test_df=pd.read_csv(r"C:\Users\benja\sysreviewfall2019\Data\all_data.csv").head(500)


def worksaver(*out_variables):
    def decorator(func):
        def wrapper(*args, **kwargs):
           if not all([var in globals() for var in out_variables]):
               print(f'Performing {func}')
               return func(*args, **kwargs)
           else:
               config = dict(globals(  ))
               if len(out_variables)>1:
                   return (config.get(var, None) for var in out_variables)
               else:
                   return config.get(out_variables[0])
        return wrapper
    return decorator

def run(loaddf):
    print('Shape of dataFrame:    ', loaddf.shape)
    df, _, _, author_globals=authors_326.main(loaddf)
    print (df[df['AF'].str.contains('Aamand')]['AF'].tolist())
    print('Shape of dataFrame:    ', df.shape)
    keep_df, not_refs=find_cites.main(df)
    keep_df, _, _, author_globals=authors_326.main(keep_df)
    new_authors=authors_326.list_all_authors(keep_df['AF'])
    print('Shape of dataFrame:    ', keep_df.shape)
    other_pubs_of_authors, left_over=find_cites.dfSortMatchList(new_authors, not_refs, 'AF' ).get_matches(True)
    keep_df=keep_df.append(other_pubs_of_authors)
    print('Shape of dataFrame:    ', keep_df.shape)
    keep_df, a_list, next_matches, extra_data=authors_326.main(keep_df)
    return keep_df, a_list, next_matches, {k: v for k, v in globals().items()}    




def test(test_df):
    return run(test_df)

@worksaver('loaddf')
def load_data():
    df= pd.read_csv(r"C:\Users\benja\sysreviewfall2019\Data\all_data.csv")
    return assign_topic_cols.main(df)
    


if __name__=='__main__':
    loaddf=load_data()
    keep_df, a_list, matches, other_outputs=run(loaddf)
    authors_326.save_results(matches, keep_df, a_list)
    