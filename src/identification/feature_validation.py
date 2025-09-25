import matplotlib.pyplot as plt
import pandas as pd
import os
from statistics import mean
import numpy as np
from utils import unpack_feature_groups
from config import DATA_DIRECTORY
import collections

class FeatureValidator:
    def __init__(self) -> None:
        self.processed_data_directory = DATA_DIRECTORY
        self.plot_data = collections.defaultdict(dict) # feature : [{device: val}]
        self.populate_plot_data()

    def check_variance(self):
        variances = {}
        for feature, device_dict in self.plot_data.items():
            device_means = [float(val) for val in device_dict.values() if pd.notna(val)]
            variances[feature] = np.var(device_means)
        return variances

    def check_feature_stability(self):
        feature_scores = {}
        for feature, device_dict in self.plot_data.items():
            device_means = [
                float(val) for val in device_dict.values() if pd.notna(val)
            ]
            if len(device_means) > 1:
                mean_val = np.mean(device_means)
                std_val = np.std(device_means)
                if mean_val != 0:
                    cv = std_val / mean_val   # relative variation
                else:
                    cv = 0.0
                feature_scores[feature] = cv
            else:
                feature_scores[feature] = 0.0
        return feature_scores

    def populate_plot_data(self):
        for file in os.listdir(self.processed_data_directory):
            df = pd.read_csv(f"{self.processed_data_directory}/{file}")
            device_name = file.split(".csv")[0]
            for column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
                all_numeric = df[column].notna().all()
                if all_numeric:
                    self.plot_data[column][device_name] = mean(df[column].tolist())
    
    def plot_values(self):
        for feature, data in self.plot_data.items():
            fig, ax = plt.subplots()
            devices = list(data.keys())
            values = list(data.values())
            ax.legend(title=feature)
            ax.bar(devices, values)
            plt.show(block = True)

    def plot_feature(self, feature):
        data = self.plot_data[feature]
        fig, ax = plt.subplots()
        devices = list(data.keys())
        values = list(data.values())
        ax.legend(title=feature)
        ax.bar(devices, values)
        plt.show(block = True)

    def find_correlation(self, feature1, feature2):
        var1 = list(self.plot_data[feature1].values())
        var2 = list(self.plot_data[feature2].values())
        x = np.array(var1, dtype=float)
        y = np.array(var2, dtype=float)

        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            return float('nan')

        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        return corr

    def correlate_all(self):
        for feature1 in self.features:
            for feature2 in self.features:
                if feature1 == feature2:
                    continue
                corr = self.find_correlation(feature1, feature2)
                print(corr, feature1, feature2)

def main():
    validator = FeatureValidator()
    valid_features = set()

    # find stability of each feature
    feature_scores = validator.check_feature_stability()
    for feature, score in feature_scores.items():
        if score > 0.05:
            valid_features.add(feature)

    # find correlated features in groups
    feature_groups = unpack_feature_groups()
    for feature_group in feature_groups:
        for i in range(1, len(feature_group)):
            correlation = validator.find_correlation(feature_group[i - 1], feature_group[i])
            if correlation >= 0.9 and feature_group[i] in valid_features:
                valid_features.remove(feature_group[i])

    print(valid_features)

if __name__ == "__main__":
    main()






        


