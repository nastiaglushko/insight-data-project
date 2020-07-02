from pathlib import Path

insight_dir = Path.home() / 'Desktop/insight_fellows/insight-project/insight-data-project/'
#insight_dir = Path.home() # for ubuntu machine

icnale_dir = insight_dir / 'data/raw/INCNALE_SM2.0_PlainText_Transcripts/'
gachon_dir = insight_dir / 'data/raw/'
subtitle_dir = insight_dir / 'data/raw/SubIMDB_All_Individual/subtitles/'

processed_data_dir = insight_dir / 'data/processed/'

model_dir = insight_dir / 'movielingo/movielingo/'