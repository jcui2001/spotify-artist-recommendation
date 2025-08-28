from UserArtistRetriever import UserArtistRetriever
import implicit
import scipy
from typing import Tuple, List
from pathlib import Path


class ImplicitRecommender:
    """The ImplicitRecommender class computes recommendations for a given user
    using the implicit library.

    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - implicit_model: an implicit model
    """

    def __init__(
        self,
        artist_retriever: UserArtistRetriever,
        implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix = None) -> None:
        """Fit the model to the user artists matrix."""
        if user_artists_matrix:
            self.implicit_model.fit(user_artists_matrix)
        elif UserArtistRetriever.get_user_artist_df():
            self.implicit_model.fit(UserArtistRetriever.get_user_artist_df())
        else:
            raise ValueError("No array-like matrix was given.")

    def recommend(
        self,
        user_id: int,
        user_artists_matrix: scipy.sparse.csr_matrix,
        n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given user."""
        # Run 'recommend' given single user_id, user-artist matrix, and top n recommendations
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_artists_matrix[user_id], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores


if __name__ == "__main__":
    
    # Initialize UserArtistRetriever
    user_artist_retriever = UserArtistRetriever()
    
    user_artists_df = user_artist_retriever.load_user_artists(Path("./data/user_artists.dat"))

    artists_df = user_artist_retriever.load_artists(Path("../lastfmdata/artists.dat"))

    # instantiate ALS using implicit
    implict_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )

    # instantiate recommender, fit, and recommend
    recommender = ImplicitRecommender(user_artist_retriever, implict_model)
    recommender.fit(user_artists_df)
    artists, scores = recommender.recommend(2, user_artists_df, n=5)

    # print results
    for artist, score in zip(artists, scores):
        print(f"{artist}: {score}")