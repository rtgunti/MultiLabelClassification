import pandas as pd
import matplotlib.pyplot as plt

_list_label_count_1 = list()


def count_number(labels):
    split_labels = str(labels).split(',')
    total_labels = len(split_labels)
    return total_labels


def change_label(row):
    if row['Eurovoc_Label'] in _list_label_count_1:
        row['Eurovoc_Label'] = 'Atomic Label'
    return row['Eurovoc_Label']


if __name__ == "__main__":
    _label_df = pd.read_csv('../data/Eurovoc_Labels.csv', encoding="latin-1")

    # distribution of class labels
    _grouped_labels_df = _label_df.groupby("Eurovoc_Label").size().reset_index(name='count')
    _sorted_grouped_labels_df = _grouped_labels_df.sort_values('count').reset_index()
    _sorted_grouped_labels_df = _sorted_grouped_labels_df.drop(columns=['index'])

    # Count number of labels present in each file
    _label_df = _label_df.groupby(['Filename'])['Eurovoc_Label'].apply(', '.join).reset_index()
    _label_df['count_of_labels'] = _label_df['Eurovoc_Label'].apply(count_number)

    # Checking maximum count of label present
    count_max_label = _label_df['count_of_labels'].max()
    index_max = _label_df['count_of_labels'].idxmax()
    print(count_max_label)
    print(_label_df['Filename'].iloc[index_max])

    # Group by count_of_labels and create a new data-frame
    _group_by_labels_df = _label_df.groupby('count_of_labels')['Filename'].apply(', '.join).reset_index()
    _group_by_labels_df['count_of_Filename'] = _group_by_labels_df['Filename'].apply(count_number)
    _group_by_labels_df = _group_by_labels_df.sort_values('count_of_Filename').reset_index()
    _group_by_labels_df = _group_by_labels_df.drop(columns='Filename')
    print(_group_by_labels_df)

    # Number of files vs Number of labels
    plt.bar(_group_by_labels_df['count_of_labels'], _group_by_labels_df['count_of_Filename'])
    plt.xlabel('Number of Labels')
    plt.ylabel('Number of Filename')
    plt.show()
