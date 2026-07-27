# LogRouter Degerlendirme Veri Seti - 20 Sorulu Mini Pilot

Hocanin talebi uzerine 116 sorulu Faz 1 pilotundan, 10 LogHub veri kumesinin tamamini kapsayacak sekilde 20 soruluk kucuk bir baslangic seti secildi: 14 easy + 4 medium + 2 hard (orijinal pilotun ~70/20/10 oranina yakin).

## Hard (2)

**[HDFS] hdfs_v1_hard_block_lifecycle_compare**
- Soru: Compare the lifecycle events recorded for block blk_-4411589101766563890 and block blk_-7029628814943626474. Correlate the sequence of events for each block and explain whether either one shows an unusual replication/termination pattern.
- Cevap: The lifecycle of block blk_‑4411589101766563890 shows a clean termination after the DataNode’s PacketResponder announces that it is terminating for this block, followed by a successful receipt event from the same node at 08:11:11 050334. In contrast, block blk_‑7029628814943626474 exhibits an abnormal pattern where the DataNode receives the block (as recorded by PacketResponder) but immediately logs a warning about an exception while serving it to another client at 08:11:10 233954. This discrepancy indicates that the first block completed its replication normally, whereas the second encountered a failure during replication or transfer. The root‑cause hypothesis is that the second block suffered a network or service interruption on the destination DataNode (or client) causing an exception during block serving, leading to incomplete replication and termination of the packet responder.
- routing_path: semantic | task: Correlation | review_status: in_review

**[LINUX] linux_v1_hard_brute_force_compare**
- Soru: Compare the SSH authentication attempts coming from 150.183.249.110 and 207.243.167.114: contrast their volume and timing. Which source looks more anomalous, and what is your root-cause hypothesis?
- Cevap: The logs show that the source IP 150.183.249.110 generated four authentication‑failure entries within a span of only two seconds (Jul 10 16:01:43–16:01:45), whereas 207.243.167.114 produced four failures over roughly eight minutes (Jul 26 07:02:27–07:02:45). This contrast in volume and timing makes the first source appear far more anomalous, as it exhibits a rapid burst of failed SSH attempts that could indicate an automated brute‑force or credential‑spraying campaign. The second source’s slower, spaced‑out pattern is consistent with manual or low‑intensity probing rather than a coordinated attack. A plausible root‑cause hypothesis is that the first IP belongs to a compromised botnet or scripted scanner attempting mass password guessing against the SSH service, while the second may be an isolated user or misconfigured client experiencing intermittent connectivity issues.
- routing_path: semantic | task: RootCauseAnalysis | review_status: in_review

## Medium (4)

**[BGL] bgl_v1_semantic_instruction_cache_parity_error_corrected_0**
- Soru: Looking at these BGL log lines, what happened and what does it mean?
- Cevap: The excerpt shows multiple instances of corrected instruction cache parity errors and double-hummer alignment exceptions occurring in rapid succession across different processors and time stamps on June 3 2005, indicating a pattern of hardware‑level memory or execution anomalies being logged by the RAS kernel. The final line also records a specific CE (correctable error) event with a masked address, confirming that correctable errors are being detected and reported in real time.
- routing_path: semantic | task: Summarization | review_status: in_review

**[MAC] mac_v1_semantic_thermal_pressure_0**
- Soru: Looking at these Mac log lines, what happened and what does it mean?
- Cevap: The system logged a thermal pressure state of 1 indicating elevated heat while memory pressure was normal and later recorded a wake event triggered by scan offload activity. These entries show normal power management transitions including sleep/wake cycles and network interface behavior without any critical errors or anomalies.
- routing_path: semantic | task: Summarization | review_status: in_review

**[OPENSTACK] openstack_v1_semantic_nova_osapi_compute_wsgi_server_0**
- Soru: Looking at these OpenStack log lines, what happened and what does it mean?
- Cevap: The log shows repeated successful API requests to list servers ending in status 200, followed by a compute node recording that instance **b9000564‑fe1a‑409b‑b8cc‑1e88b294cd1d** experienced a "VM Started" lifecycle event and then immediately a "VM Paused" event. This indicates the virtual machine was launched and then paused shortly thereafter.
- routing_path: semantic | task: Summarization | review_status: in_review

**[ZOOKEEPER] zookeeper_v1_semantic_fastleaderelection_0**
- Soru: Looking at these Zookeeper log lines, what happened and what does it mean?
- Cevap: The log shows repeated connection failures and restarts of a quorum election process on server myid=1, indicating it is losing connections to peers and attempting to re-establish them. The warnings about SendWorker leaving threads and RecvWorker reporting broken connections suggest the server is experiencing instability in its communication channels with other Zookeeper ensemble members.
- routing_path: semantic | task: Summarization | review_status: in_review

## Easy (14)

**[APACHE] apache_v1_count_notice_0**
- Soru: How many log lines contain '[notice]'?
- Cevap: 1405
- routing_path: sql | task: Aggregation | review_status: verified

**[BGL] bgl_v1_count_double_hummer_alignment_exceptions_0**
- Soru: How many log lines contain 'double-hummer alignment exceptions'?
- Cevap: 109
- routing_path: sql | task: Aggregation | review_status: verified

**[BGL] bgl_v1_presence_kerndtlb_0**
- Soru: Does the log contain any line with 'KERNDTLB'?
- Cevap: Yes
- routing_path: sql | task: Aggregation | review_status: verified

**[HADOOP] hadoop_v1_count_container_0**
- Soru: How many log lines contain 'Container'?
- Cevap: 547
- routing_path: sql | task: Aggregation | review_status: verified

**[HDFS] hdfs_v1_count_packetresponder_0**
- Soru: How many log lines contain 'PacketResponder'?
- Cevap: 603
- routing_path: sql | task: Aggregation | review_status: verified

**[HDFS] hdfs_v1_presence_exception_0**
- Soru: Does the log contain any line with 'Exception'?
- Cevap: Yes
- routing_path: sql | task: Aggregation | review_status: verified

**[LINUX] linux_v1_count_authentication_failure_0**
- Soru: How many log lines contain 'authentication failure'?
- Cevap: 490
- routing_path: sql | task: Aggregation | review_status: verified

**[LINUX] linux_v1_presence_invalid_user_0**
- Soru: Does the log contain any line with 'Invalid user'?
- Cevap: No
- routing_path: sql | task: Aggregation | review_status: verified

**[MAC] mac_v1_count_kernel_0**
- Soru: How many log lines contain 'kernel'?
- Cevap: 775
- routing_path: sql | task: Aggregation | review_status: verified

**[OPENSSH] openssh_v1_count_failed_password_0**
- Soru: How many log lines contain 'Failed password'?
- Cevap: 520
- routing_path: sql | task: Aggregation | review_status: verified

**[OPENSSH] openssh_v1_presence_possible_break_in_attempt_0**
- Soru: Does the log contain any line with 'POSSIBLE BREAK-IN ATTEMPT'?
- Cevap: Yes
- routing_path: sql | task: Aggregation | review_status: verified

**[OPENSTACK] openstack_v1_count_status_200_0**
- Soru: How many log lines contain 'status: 200'?
- Cevap: 933
- routing_path: sql | task: Aggregation | review_status: verified

**[WINDOWS] windows_v1_count_cbs_0**
- Soru: How many log lines contain 'CBS'?
- Cevap: 1973
- routing_path: sql | task: Aggregation | review_status: verified

**[ZOOKEEPER] zookeeper_v1_count_warn_0**
- Soru: How many log lines contain 'WARN'?
- Cevap: 1318
- routing_path: sql | task: Aggregation | review_status: verified
