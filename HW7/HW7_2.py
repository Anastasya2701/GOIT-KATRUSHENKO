import numpy as np 
import zipfile
import pandas as pd 

def extract_zip(file_path, extract_to='.'):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_movielens_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    n_users = data['user_id'].nunique()
    n_items = data['item_id'].nunique()

    R = np.zeros((n_users, n_items))
    for row in data.itertuples():
        R[row.user_id - 1, row.item_id - 1] = row.rating
    return R

def load_movie_titles(file_path):
    movie_titles = {}
    with open(file_path, 'r', encoding='latin-1') as file:
        for line in file:
            parts = line.strip().split(' ', 1)
            movie_id = int(parts[0])
            title = parts[1]
            movie_titles[movie_id] = title
    return movie_titles

def initialize_matrices(n_users, n_items, n_factors):
    U = np.random.normal(scale=1./n_factors, size=(n_users, n_factors))
    V = np.random.normal(scale=1./n_factors, size=(n_items, n_factors))
    return U, V

def loss_function(R, U, V, lambda_reg):
    predicted_ratings = U.dot(V.T)
    mask = R > 0
    error = (R[mask] - predicted_ratings[mask])**2
    loss = np.sum(error) + lambda_reg * (np.sum(U**2) + np.sum(V**2))
    return loss

def gradient_descent(R, U, V, learning_rate, lambda_reg, n_epochs):
    n_users, n_items = R.shape
    for epoch in range(n_epochs):
        for i in range(n_users):
            for j in range(n_items):
                if R[i, j] > 0:
                    error = R[i, j] - np.dot(U[i, :], V[j, :].T)
                    U[i, :] += learning_rate * (error * V[j, :] - lambda_reg * U[i, :])
                    V[j, :] += learning_rate * (error * U[i, :] - lambda_reg * V[j, :])


        loss = loss_function(R, U, V, lambda_reg)
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}')
    return U, V

def predict_ratings(U, V):
    return U.dot(V.T)

if __name__ == "__main__":
    extract_zip('C:/Users/Dell/Documents/My_repo/goit-KATRUSHENKO/HW7/ml-100k.zip', './ml-100k')

    rating_file = 'C:/Users/Dell/Documents/My_repo/goit-KATRUSHENKO/HW7/ml-100k/ml-100k/u.data'
    movie_titles_file = 'C:/Users/Dell/Documents/My_repo/goit-KATRUSHENKO/HW7/movie_ids.txt'

    R = load_movielens_data(rating_file)
    movie_titles = load_movie_titles(movie_titles_file)

    n_users, n_items = R.shape
    print(f'Матриця рейтингів: {R.shape}')

    n_factors = 10
    U, V = initialize_matrices(n_users, n_items, n_factors)

    learning_rate = 0.01
    lambda_reg = 0.1
    n_epochs = 20

    U, V = gradient_descent(R, U, V, learning_rate, lambda_reg, n_epochs)

    predict_ratings = predict_ratings(U, V)

    user_id = 0
    print(f'\nПрогнозовані рейтинги для користувача {user_id+1}:')
    for item_id, rating in enumerate(predict_ratings[user_id]):
        if R[user_id, item_id] == 0:
            print(f'{movie_titles.get(item_id+1, 'Невідомий фільм')}: {rating:.2f}')