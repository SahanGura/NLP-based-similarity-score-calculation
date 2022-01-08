import argparse
import json
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
from scipy import spatial

class Dataset:
    def __init__(self,file_name,dataset={}):
        self.file_name = file_name
        self.dataset = dataset

    def load(self):
        with open(self.file_name) as json_file:
            dataset = json.load(json_file)
            self.skill_discs = list(dataset.values())
            self.skills = list(dataset.keys()) 

    def write_json(self):
        with open(self.file_name, 'w') as fp:
            # json.dump(dataset, fp)
            fp.write(json.dumps(self.dataset, indent=4))
            print("Dataset save to: " + self.file_name)

    def create_df(self):
        #Preparing the dataframe
        sim_df = pd.DataFrame(columns=self.skills)
        key_dict = {"key":self.skills}
        key_df = pd.DataFrame(key_dict)
        data_df = pd.concat([key_df,sim_df],axis=1)
        data_df = data_df.set_index('key')

        # import itertools
        # lst = list(itertools.repeat(0, len(skills)))
        # # print(lst)
        # for key in skills:
        #   data_df.loc[key] = lst
        # display(data_df)

        #Calculating similarity scores
        sent_embeddings = model.encode(self.skill_discs)
        idx=0
        for key_embedding in sent_embeddings:
            sim_score_lst = []
            for embedding in sent_embeddings:
                sim_score = 1 - spatial.distance.cosine(key_embedding, embedding)
                sim_score_lst.append(sim_score)
            data_df.loc[self.skills[idx]] = sim_score_lst
            idx+=1
        return data_df

class TopSkills:
    def __init__(self,data_df,n_skills=6,file_name=''):
        self.data_df = data_df
        self.n_skills = n_skills
        self.file_name = file_name

    def filt_top_skills(self):
        #changing data type of dataframe values
        self.data_df[self.data_df.columns] = self.data_df[self.data_df.columns].astype(float)
        # print(data_df.dtypes)

        self.sim_skills_dict = {}
        for col in self.data_df.columns:
            maxValueIndexObj = self.data_df[col].nlargest(n=self.n_skills)
            self.sim_skills_dict[col] = list(maxValueIndexObj.index)[1:]

        return self.sim_skills_dict

    def write_top_skills(self):
        with open(self.file_name, 'w') as fp1:
            # json.dump(sim_skills_dict, fp1)
            fp1.write(json.dumps(self.sim_skills_dict, indent=4))

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument('path_to_input', type=str, help='path to input file')
    parser.add_argument('path_to_output', type=str, help='path to output file')
    parser.add_argument('n_skills', type=int, help='num_skills=6')
    # Parse the argument
    args = parser.parse_args()


    dataset1 = Dataset(args.path_to_input)
    dataset1.load()
    data_df = dataset1.create_df()

    TopSkills1 = TopSkills(data_df,n_skills=args.n_skills,file_name=args.path_to_output)
    sim_skills_dict = TopSkills1.filt_top_skills()
    TopSkills1.write_top_skills()
    
