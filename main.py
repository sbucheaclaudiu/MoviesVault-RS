import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)


# Funcție pentru calculul ratingului ponderat
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


# Funcție pentru recomandări îmbunătățite
def improved_recommendations(title, smd, cosine_sim, C, m):
    indices = pd.Series(smd.index, index=smd['title'])
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')

    qualified = movies.loc[
        (movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())].copy()
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)

    return qualified['title'].tolist()



@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    print(title)
    if title is None:
        return jsonify({'error': 'No title provided'}), 400

    try:
        results = improved_recommendations(title, smd, cosine_sim, C, m)
        return jsonify({'recommendations': results})
    except Exception as e:
        print(str(e))
        return jsonify({'error': 'Title not found in database'}), 404


if __name__ == "__main__":
    smd = pd.read_csv('smd_processed.csv')
    cosine_sim = joblib.load('cosine_similarity_model.pkl')
    C = smd['vote_average'].mean()
    m = smd['vote_count'].quantile(0.90)
    app.run(debug=True)
