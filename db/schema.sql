PRAGMA foreign_keys = ON;

CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  username TEXT,
  display_name TEXT
);

CREATE UNIQUE INDEX idx_users_username ON users(username);

CREATE TABLE tweets (
  id INTEGER PRIMARY KEY,
  user_id INTEGER,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL,
  in_reply_to_tweet_id INTEGER,
  retweet_of_tweet_id INTEGER,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL,
  FOREIGN KEY(in_reply_to_tweet_id) REFERENCES tweets(id) ON DELETE SET NULL,
  FOREIGN KEY(retweet_of_tweet_id) REFERENCES tweets(id) ON DELETE SET NULL
);

CREATE INDEX idx_tweets_created_at ON tweets(created_at);
CREATE INDEX idx_tweets_user_id ON tweets(user_id);
CREATE INDEX idx_tweets_retweet_of ON tweets(retweet_of_tweet_id);

CREATE TABLE hashtags (
  id INTEGER PRIMARY KEY,
  tag TEXT NOT NULL UNIQUE
);

CREATE TABLE tweet_hashtags (
  tweet_id INTEGER NOT NULL,
  hashtag_id INTEGER NOT NULL,
  PRIMARY KEY(tweet_id, hashtag_id),
  FOREIGN KEY(tweet_id) REFERENCES tweets(id) ON DELETE CASCADE,
  FOREIGN KEY(hashtag_id) REFERENCES hashtags(id) ON DELETE CASCADE
);

CREATE TABLE urls (
  id INTEGER PRIMARY KEY,
  url TEXT NOT NULL UNIQUE
);

CREATE TABLE tweet_urls (
  tweet_id INTEGER NOT NULL,
  url_id INTEGER NOT NULL,
  PRIMARY KEY(tweet_id, url_id),
  FOREIGN KEY(tweet_id) REFERENCES tweets(id) ON DELETE CASCADE,
  FOREIGN KEY(url_id) REFERENCES urls(id) ON DELETE CASCADE
);

CREATE TABLE tweet_mentions (
  tweet_id INTEGER NOT NULL,
  mentioned_user_id INTEGER NOT NULL,
  PRIMARY KEY(tweet_id, mentioned_user_id),
  FOREIGN KEY(tweet_id) REFERENCES tweets(id) ON DELETE CASCADE,
  FOREIGN KEY(mentioned_user_id) REFERENCES users(id) ON DELETE CASCADE
);
