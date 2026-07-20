-- verify.sql
-- output/pilot/questions.json'daki 20 sorunun cevaplarini bagimsiz olarak
-- SQL ile dogrulamak icin. Veri, sql_verify container'i ilk kez ayaga
-- kalkarken init/01_load.sql tarafindan otomatik yuklendi (raw_logs tablosu).
--
-- Eslesme mantigi projedeki ile birebir ayni: case-insensitive substring
-- arama (bkz. datasetgen/question_generators.py::_count_matches).
--
-- Calistirma:
--   docker compose exec sql_verify psql -U postgres -d logs -f /dev/stdin < sql_verification/verify.sql
-- ya da interaktif:
--   docker compose exec -it sql_verify psql -U postgres -d logs

-- linux_v1_count_authentication_failure_0 -- beklenen: 490
SELECT COUNT(*) FROM raw_logs WHERE dataset='linux' AND line ILIKE '%authentication failure%';

-- linux_v1_presence_invalid_user_0 -- beklenen: No (0 satir)
SELECT COUNT(*) > 0 AS present FROM raw_logs WHERE dataset='linux' AND line ILIKE '%Invalid user%';

-- apache_v1_count_notice_0 -- beklenen: 1405
SELECT COUNT(*) FROM raw_logs WHERE dataset='apache' AND line ILIKE '%[notice]%';

-- apache_v1_presence_mod_jk_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM raw_logs WHERE dataset='apache' AND line ILIKE '%mod_jk%';

-- windows_v1_count_cbs_0 -- beklenen: 1973
SELECT COUNT(*) FROM raw_logs WHERE dataset='windows' AND line ILIKE '%CBS%';

-- windows_v1_presence_error_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM raw_logs WHERE dataset='windows' AND line ILIKE '%Error%';

-- mac_v1_count_kernel_0 -- beklenen: 775
SELECT COUNT(*) FROM raw_logs WHERE dataset='mac' AND line ILIKE '%kernel%';

-- mac_v1_presence_thermal_pressure_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM raw_logs WHERE dataset='mac' AND line ILIKE '%Thermal pressure%';

-- hdfs_v1_count_packetresponder_0 -- beklenen: 603
SELECT COUNT(*) FROM raw_logs WHERE dataset='hdfs' AND line ILIKE '%PacketResponder%';

-- hdfs_v1_presence_exception_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM raw_logs WHERE dataset='hdfs' AND line ILIKE '%Exception%';

-- openssh_v1_count_failed_password_0 -- beklenen: 520
SELECT COUNT(*) FROM raw_logs WHERE dataset='openssh' AND line ILIKE '%Failed password%';

-- openssh_v1_presence_possible_break_in_attempt_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM raw_logs WHERE dataset='openssh' AND line ILIKE '%POSSIBLE BREAK-IN ATTEMPT%';

-- bgl_v1_count_double_hummer_alignment_exceptions_0 -- beklenen: 109
SELECT COUNT(*) FROM raw_logs WHERE dataset='bgl' AND line ILIKE '%double-hummer alignment exceptions%';

-- bgl_v1_presence_kerndtlb_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM raw_logs WHERE dataset='bgl' AND line ILIKE '%KERNDTLB%';

-- hadoop_v1_count_container_0 -- beklenen: 547
SELECT COUNT(*) FROM raw_logs WHERE dataset='hadoop' AND line ILIKE '%Container%';

-- hadoop_v1_presence_mrappmaster_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM raw_logs WHERE dataset='hadoop' AND line ILIKE '%MRAppMaster%';

-- zookeeper_v1_count_warn_0 -- beklenen: 1318
SELECT COUNT(*) FROM raw_logs WHERE dataset='zookeeper' AND line ILIKE '%WARN%';

-- zookeeper_v1_presence_exception_0 -- beklenen: Yes
SELECT COUNT(*) > 0 AS present FROM raw_logs WHERE dataset='zookeeper' AND line ILIKE '%Exception%';

-- openstack_v1_count_status_200_0 -- beklenen: 933
SELECT COUNT(*) FROM raw_logs WHERE dataset='openstack' AND line ILIKE '%status: 200%';

-- openstack_v1_presence_error_0 -- beklenen: No (0 satir)
SELECT COUNT(*) > 0 AS present FROM raw_logs WHERE dataset='openstack' AND line ILIKE '%ERROR%';
