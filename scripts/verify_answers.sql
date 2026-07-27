-- scripts/verify_answers.sql
-- output/pilot/questions.json'daki 20 sorunun cevaplarini bagimsiz olarak
-- SQL ile dogrulamak icin. Veri, loghub servisinin kendi Postgres'inde
-- zaten yuklu (fetch_corpus.py::load_into_postgres, container her acilista
-- calisir) -- ayri bir dogrulama veritabani yok.
--
-- Eslesme mantigi projedeki ile birebir ayni: case-insensitive substring
-- arama (bkz. src/generators/easy_tier.py::count_matches /
-- src/utils/helper_postgres.py::count_literal).
--
-- Calistirma (POSTGRES_USER/POSTGRES_DB .env'deki degerlerle ayni olmali):
--   docker compose -f docker/compose.yml exec -T loghub psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" < scripts/verify_answers.sql
-- ya da interaktif:
--   docker compose -f docker/compose.yml exec -it loghub psql -U loghub -d loghub

-- linux_v1_count_authentication_failure_0 -- beklenen: 490
SELECT COUNT(*) FROM lines WHERE dataset='linux' AND text ILIKE '%authentication failure%';

-- linux_v1_presence_invalid_user_0 -- beklenen: No (0 satir)
SELECT COUNT(*) > 0 AS present FROM lines WHERE dataset='linux' AND text ILIKE '%Invalid user%';

-- apache_v1_count_notice_0 -- beklenen: 1405
SELECT COUNT(*) FROM lines WHERE dataset='apache' AND text ILIKE '%[notice]%';

-- apache_v1_presence_mod_jk_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM lines WHERE dataset='apache' AND text ILIKE '%mod_jk%';

-- windows_v1_count_cbs_0 -- beklenen: 1973
SELECT COUNT(*) FROM lines WHERE dataset='windows' AND text ILIKE '%CBS%';

-- windows_v1_presence_error_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM lines WHERE dataset='windows' AND text ILIKE '%Error%';

-- mac_v1_count_kernel_0 -- beklenen: 775
SELECT COUNT(*) FROM lines WHERE dataset='mac' AND text ILIKE '%kernel%';

-- mac_v1_presence_thermal_pressure_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM lines WHERE dataset='mac' AND text ILIKE '%Thermal pressure%';

-- hdfs_v1_count_packetresponder_0 -- beklenen: 603
SELECT COUNT(*) FROM lines WHERE dataset='hdfs' AND text ILIKE '%PacketResponder%';

-- hdfs_v1_presence_exception_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM lines WHERE dataset='hdfs' AND text ILIKE '%Exception%';

-- openssh_v1_count_failed_password_0 -- beklenen: 520
SELECT COUNT(*) FROM lines WHERE dataset='openssh' AND text ILIKE '%Failed password%';

-- openssh_v1_presence_possible_break_in_attempt_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM lines WHERE dataset='openssh' AND text ILIKE '%POSSIBLE BREAK-IN ATTEMPT%';

-- bgl_v1_count_double_hummer_alignment_exceptions_0 -- beklenen: 109
SELECT COUNT(*) FROM lines WHERE dataset='bgl' AND text ILIKE '%double-hummer alignment exceptions%';

-- bgl_v1_presence_kerndtlb_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM lines WHERE dataset='bgl' AND text ILIKE '%KERNDTLB%';

-- hadoop_v1_count_container_0 -- beklenen: 547
SELECT COUNT(*) FROM lines WHERE dataset='hadoop' AND text ILIKE '%Container%';

-- hadoop_v1_presence_mrappmaster_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM lines WHERE dataset='hadoop' AND text ILIKE '%MRAppMaster%';

-- zookeeper_v1_count_warn_0 -- beklenen: 1318
SELECT COUNT(*) FROM lines WHERE dataset='zookeeper' AND text ILIKE '%WARN%';

-- zookeeper_v1_presence_exception_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM lines WHERE dataset='zookeeper' AND text ILIKE '%Exception%';

-- openstack_v1_count_status_200_0 -- beklenen: 933
SELECT COUNT(*) FROM lines WHERE dataset='openstack' AND text ILIKE '%status: 200%';

-- openstack_v1_presence_error_0 -- beklenen: No (0 satir)
SELECT COUNT(*) > 0 AS present FROM lines WHERE dataset='openstack' AND text ILIKE '%ERROR%';
