from pathlib import Path
import pandas as pd
import scipy


class UserArtistRetriever:
    """The ArtistRetriever class gets the artist name from the artist ID."""

    def __init__(self):
        self._artists_df = None
        self._user_artists_df = None
    
    def get_user_artist_df(self):
        return self.__user_artists_df

    def get_artist_name_from_id(self, artist_id: int) -> str:
        """Return the artist name from the artist ID."""
        return self._artists_df.loc[artist_id, "name"]

    def load_artists(self, artists_file: Path) -> None:
        """Load the artists file and store it as a Pandas DataFrame."""
        artists_df = pd.read_csv(artists_file, sep="\t")
        artists_df = artists_df.set_index("id")
        self._artists_df = artists_df
        
        return artists_df
    
    def load_user_artists(self, user_artists_file: Path) -> scipy.sparse.csr_matrix:
        """Load the user artists file and return a user-artist matrix in csr format."""
        user_artists = pd.read_csv(user_artists_file, sep="\t")
        user_artists.set_index(["userID", "artistID"], inplace=True)
        coo = scipy.sparse.coo_matrix(
            (
                user_artists.weight.astype(float),
                (
                    user_artists.index.get_level_values(0),
                    user_artists.index.get_level_values(1),
                ),
            )
        )
        
        # Load sparse ataframe into Compressed Sparse Row Format
        self._user_artists_df = coo.tocsr()
        
        return coo.tocsr()