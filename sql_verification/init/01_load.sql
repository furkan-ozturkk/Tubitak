-- 01_load.sql
-- Postgres resmi imajinin otomatik-init mekanizmasi sayesinde bu dosya,
-- container ilk kez olusturuldugunda (veri dizini bossa) OTOMATIK calisir.
-- Log dosyalarini raw_logs tablosuna yukler. Ayni mantik verify.sql'de de
-- var; burada sadece yol container icindeki mount noktasina gore (/logs).

CREATE TABLE raw_logs (dataset text, line text);
CREATE TEMP TABLE staging (line text);

\copy staging(line) FROM '/logs/Linux_2k.log' WITH (FORMAT csv, DELIMITER E'\x01', QUOTE E'\x02', ESCAPE E'\x02')
INSERT INTO raw_logs SELECT 'linux', line FROM staging; TRUNCATE staging;

\copy staging(line) FROM '/logs/Apache_2k.log' WITH (FORMAT csv, DELIMITER E'\x01', QUOTE E'\x02', ESCAPE E'\x02')
INSERT INTO raw_logs SELECT 'apache', line FROM staging; TRUNCATE staging;

\copy staging(line) FROM '/logs/Windows_2k.log' WITH (FORMAT csv, DELIMITER E'\x01', QUOTE E'\x02', ESCAPE E'\x02')
INSERT INTO raw_logs SELECT 'windows', line FROM staging; TRUNCATE staging;

\copy staging(line) FROM '/logs/Mac_2k.log' WITH (FORMAT csv, DELIMITER E'\x01', QUOTE E'\x02', ESCAPE E'\x02')
INSERT INTO raw_logs SELECT 'mac', line FROM staging; TRUNCATE staging;

\copy staging(line) FROM '/logs/HDFS_2k.log' WITH (FORMAT csv, DELIMITER E'\x01', QUOTE E'\x02', ESCAPE E'\x02')
INSERT INTO raw_logs SELECT 'hdfs', line FROM staging; TRUNCATE staging;

\copy staging(line) FROM '/logs/OpenSSH_2k.log' WITH (FORMAT csv, DELIMITER E'\x01', QUOTE E'\x02', ESCAPE E'\x02')
INSERT INTO raw_logs SELECT 'openssh', line FROM staging; TRUNCATE staging;

\copy staging(line) FROM '/logs/BGL_2k.log' WITH (FORMAT csv, DELIMITER E'\x01', QUOTE E'\x02', ESCAPE E'\x02')
INSERT INTO raw_logs SELECT 'bgl', line FROM staging; TRUNCATE staging;

\copy staging(line) FROM '/logs/Hadoop_2k.log' WITH (FORMAT csv, DELIMITER E'\x01', QUOTE E'\x02', ESCAPE E'\x02')
INSERT INTO raw_logs SELECT 'hadoop', line FROM staging; TRUNCATE staging;

\copy staging(line) FROM '/logs/Zookeeper_2k.log' WITH (FORMAT csv, DELIMITER E'\x01', QUOTE E'\x02', ESCAPE E'\x02')
INSERT INTO raw_logs SELECT 'zookeeper', line FROM staging; TRUNCATE staging;

\copy staging(line) FROM '/logs/OpenStack_2k.log' WITH (FORMAT csv, DELIMITER E'\x01', QUOTE E'\x02', ESCAPE E'\x02')
INSERT INTO raw_logs SELECT 'openstack', line FROM staging; TRUNCATE staging;

CREATE INDEX idx_raw_logs_dataset ON raw_logs(dataset);
