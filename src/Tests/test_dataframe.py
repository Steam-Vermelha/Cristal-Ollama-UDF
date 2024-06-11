import pytest
from DataBusiness import BaseConnection


def test_dataframe_return():
    BaseConnection.charging_collections_in_dataframe()
    df_games = BaseConnection.df_games

    print(df_games.head())
def test_dataframe_duplicate():
    BaseConnection.charging_collections_in_dataframe()
    df_games = BaseConnection.df_games

    df_games = df_games.applymap(lambda x: str(x) if isinstance(x, list) else x)

    duplicatas = df_games.duplicated()
    assert not duplicatas.any(), "Existem linhas duplicadas no DataFrame"


