# LogRouter Degerlendirme Veri Seti - Faz 1 Pilot Ornek Sorular

Toplam pilot veri setinde 116 soru var; asagida zorluk seviyelerine gore dengeli secilmis 28 ornek bulunuyor (6 hard + 10 medium + 12 easy, 10 LogHub veri kumesinin tamamindan).

## Hard (cok-olay / kok-neden / sentez)

**[BGL] bgl_v1_hard_hardware_error_chain**
- Soru: Correlate the occurrences of 'double-hummer alignment exceptions' and 'instruction cache parity error corrected' across the observation window. Propose a root-cause hypothesis for why this supercomputer node is showing both error types.
- Cevap: The evidence shows that the same compute node (R02-M1-N0-C) experienced multiple instances of both double‑hummer alignment exceptions and instruction‑cache parity errors within a short time window on 2005‑06‑03, indicating that these error types co‑occur on this hardware. The repeated “double‑hummer alignment exceptions” entries (e.g., counts of 141–199 680) demonstrate frequent alignment faults in the node’s execution pipelines, while the four consecutive “instruction cache parity error corrected” messages confirm that the instruction cache is repeatedly detecting and correcting parity mismatches. Because both error streams originate from the same RAS KERNEL INFO source on identical rack identifiers (R21‑M1‑N6‑C, R24‑M0‑N1‑C, etc.) and timestamps cluster around the same minutes, it suggests a systemic fault rather than isolated glitches. The most plausible root‑cause hypothesis is that a hardware defect—such as a failing memory controller or corrupted cache tags—is causing misaligned memory accesses that trigger double‑hummer exceptions while simultaneously corrupting instruction fetches, leading the cache parity logic to correct errors repeatedly. This combined symptom pattern points to a deteriorating node-level component (e.g., faulty SRAM/DRAM module or bus interface) that degrades both alignment checking and instruction cache integrity.
- routing_path: semantic | task: RootCauseAnalysis | review_status: in_review

**[HADOOP] hadoop_v1_hard_container_task_compare**
- Soru: Correlate the task events logged for container_1445144423722_0020_01_000005 and container_1445144423722_0020_01_000003. Do their event sequences suggest a normal execution, or does one of them show signs of a task/container failure? Give your root-cause hypothesis.
- Cevap: The two container logs show a clear parallelism in their event sequences, but the timing and nature of the cleanup events reveal that one task completed normally while the other exhibits signs of abnormal termination. In Group container_1445144423722_0020_01_000005, after the container is launched on NM MSRA‑SA‑41 at 18:02:47 and the task attempt begins, a cleanup event occurs only after roughly two minutes (at 18:04:50), indicating that the container was allowed to run its full map phase before being removed. In contrast, Group container_1445144423722_0020_01_000003 shows a launch at 18:01:58 and an immediate cleanup at 18:06:28, but the interval between launch and cleanup is preceded by no successful task execution logs; instead, the only recorded events are the allocation and launch, followed directly by cleanup without any reported task attempt progress or completion messages. This abrupt transition from launch to remote cleanup in the second group suggests that the container was terminated prematurely—likely due to a failure of the NodeManager or an abnormal kill signal—preventing the map task from completing its work. Consequently, the root‑cause hypothesis is that the first container (000005) completed its map attempt successfully and was cleanly shut down, whereas the second container (000003) suffered a node‑level failure or premature termination, causing an abnormal cleanup and thus a task/container failure.
- routing_path: semantic | task: Correlation | review_status: in_review

**[HDFS] hdfs_v1_hard_block_lifecycle_compare**
- Soru: Compare the lifecycle events recorded for block blk_-4411589101766563890 and block blk_-7029628814943626474. Correlate the sequence of events for each block and explain whether either one shows an unusual replication/termination pattern.
- Cevap: The lifecycle of block blk_‑4411589101766563890 shows a clean termination after the DataNode’s PacketResponder announces that it is terminating for this block, followed by a successful receipt event from the same node at 08:11:11 050334. In contrast, block blk_‑7029628814943626474 exhibits an abnormal pattern where the DataNode receives the block (as recorded by PacketResponder) but immediately logs a warning about an exception while serving it to another client at 08:11:10 233954. This discrepancy indicates that the first block completed its replication normally, whereas the second encountered a failure during replication or transfer. The root‑cause hypothesis is that the second block suffered a network or service interruption on the destination DataNode (or client) causing an exception during block serving, leading to incomplete replication and termination of the packet responder.
- routing_path: semantic | task: Correlation | review_status: in_review

**[LINUX] linux_v1_hard_brute_force_compare**
- Soru: Compare the SSH authentication attempts coming from 150.183.249.110 and 207.243.167.114: contrast their volume and timing. Which source looks more anomalous, and what is your root-cause hypothesis?
- Cevap: The logs show that the source IP 150.183.249.110 generated four authentication‑failure entries within a span of only two seconds (Jul 10 16:01:43–16:01:45), whereas 207.243.167.114 produced four failures over roughly eight minutes (Jul 26 07:02:27–07:02:45). This contrast in volume and timing makes the first source appear far more anomalous, as it exhibits a rapid burst of failed SSH attempts that could indicate an automated brute‑force or credential‑spraying campaign. The second source’s slower, spaced‑out pattern is consistent with manual or low‑intensity probing rather than a coordinated attack. A plausible root‑cause hypothesis is that the first IP belongs to a compromised botnet or scripted scanner attempting mass password guessing against the SSH service, while the second may be an isolated user or misconfigured client experiencing intermittent connectivity issues.
- routing_path: semantic | task: RootCauseAnalysis | review_status: in_review

**[OPENSSH] openssh_v1_hard_attacker_compare**
- Soru: Compare the attack patterns from source 183.62.140.253 and source 187.141.143.180 (target accounts, request cadence, outcome). Which one is the more anomalous attacker, and what is your root-cause hypothesis?
- Cevap: The evidence shows that source 183.62.140.253 generated three distinct invalid‑user attempts (zhangyan, dff) within a single minute before terminating the session with “Bye Bye,” whereas source 187.141.143.180 issued two repeated failed password attempts for the privileged account root over a six‑second interval and also ended each attempt with the same disconnect message. Both groups share the identical pre‑authentication disconnection pattern, but only the second group targets a legitimate system account (root) and exhibits a tighter request cadence of exactly three seconds between failures, making its behavior more anomalous in terms of intent and timing. The root‑cause hypothesis is that an automated brute‑force script is being used to probe for weak credentials, with the first source likely testing generic username lists while the second source is specifically attempting credential theft against high‑value accounts; the consistent “Bye Bye” termination indicates a coordinated disconnect after each failed attempt, suggesting a single attacker or shared tooling orchestrating both sources.
- routing_path: semantic | task: RootCauseAnalysis | review_status: in_review

**[ZOOKEEPER] zookeeper_v1_hard_ensemble_member_compare**
- Soru: Compare the coordination activity logged for ensemble member myid=1 and myid=2. Correlate their event sequences and explain whether either member's behavior looks anomalous for a Zookeeper ensemble.
- Cevap: The logs show that myid=1 entered the LOOKING state at 23:43:23,613 after reporting a warning about being unable to open a channel to server 2 on port 3888, indicating it was trying to re‑elect but could not reach its peer. In contrast, myid=2 progressed from LOOKING (noted at 19:56:54,754 when it announced “LOOKING” as part of a FastLeaderElection) to the LEADING state in the same log entry and later performed snapshotting operations on 2015‑07‑30 and 2015‑07‑31, demonstrating that it successfully completed election and persisted its transaction log. The only anomaly is myid=1’s failure to establish a channel to server 2, which caused it to remain in LOOKING while the rest of the ensemble moved forward; this suggests that myid=1 is experiencing network or configuration problems preventing it from participating fully in quorum elections. Consequently, the root‑cause hypothesis is that a connectivity issue (e.g., firewall block or mis‑configured peer address) on myid=1 prevents it from communicating with server 2, leading to an incomplete election and leaving the ensemble unable to achieve consensus until the connection is restored.
- routing_path: semantic | task: Correlation | review_status: in_review

## Medium (tek-olay aciklama/ozet)

**[APACHE] apache_v1_semantic_child_workerenv_in_error_state_0**
- Soru: Looking at these Apache log lines, what happened and what does it mean?
- Cevap: The log shows repeated errors indicating that the Apache module mod_jk is encountering a persistent issue where its worker process named "workerEnv" repeatedly enters an error state 6, suggesting a failure in initializing or maintaining the Java connector's child processes despite successful configuration loading. This pattern points to a problem with the JK2 integration setup rather than general server operation.
- routing_path: semantic | task: Summarization | review_status: in_review

**[BGL] bgl_v1_semantic_instruction_cache_parity_error_corrected_0**
- Soru: Looking at these BGL log lines, what happened and what does it mean?
- Cevap: The excerpt shows multiple instances of corrected instruction cache parity errors and double-hummer alignment exceptions occurring in rapid succession across different processors and time stamps on June 3 2005, indicating a pattern of hardware‑level memory or execution anomalies being logged by the RAS kernel. The final line also records a specific CE (correctable error) event with a masked address, confirming that correctable errors are being detected and reported in real time.
- routing_path: semantic | task: Summarization | review_status: in_review

**[HADOOP] hadoop_v1_semantic_mrappmaster_0**
- Soru: Looking at these Hadoop log lines, what happened and what does it mean?
- Cevap: The excerpt shows that a MapReduce application master has been created and is initializing its execution environment in YARN, including registering token types and output committers for job handling. This indicates the start of a MapReduce job submission process within the Hadoop ecosystem.
- routing_path: semantic | task: Summarization | review_status: in_review

**[HDFS] hdfs_v1_semantic_addstoredblock_0**
- Soru: Looking at these HDFS log lines, what happened and what does it mean?
- Cevap: The excerpt shows NameNode updates to its block map indicating new blocks have been added by DataNodes at specific IP addresses and ports, confirming ongoing storage operations where data nodes are reporting block additions or termination of packet responses for large 64MB blocks. These messages reflect the normal flow of block management between DataNodes and the NameNode in HDFS.
- routing_path: semantic | task: Summarization | review_status: in_review

**[LINUX] linux_v1_semantic_check_pass_user_unknown_0**
- Soru: Looking at these Linux log lines, what happened and what does it mean?
- Cevap: The log shows multiple failed SSH authentication attempts for the root user from a single external IP address (218.188.2.4 and later 220-135-151-1.hinet-ip.hinet.net), with each attempt recorded as an "authentication failure" by the sshd PAM module using pam_unix, indicating repeated unsuccessful login attempts targeting the root account from a remote host.
- routing_path: semantic | task: Summarization | review_status: in_review

**[MAC] mac_v1_semantic_thermal_pressure_0**
- Soru: Looking at these Mac log lines, what happened and what does it mean?
- Cevap: The system logged a thermal pressure state of 1 indicating elevated heat while memory pressure was normal and later recorded a wake event triggered by scan offload activity. These entries show normal power management transitions including sleep/wake cycles and network interface behavior without any critical errors or anomalies.
- routing_path: semantic | task: Summarization | review_status: in_review

**[OPENSSH] openssh_v1_semantic_possible_break_in_attempt_0**
- Soru: Looking at these OpenSSH log lines, what happened and what does it mean?
- Cevap: The log shows multiple failed authentication attempts from IP 173.234.31.186 targeting a non‑existent user “webmaster”, including reverse‑DNS lookup failures and explicit “Invalid user” messages, indicating a brute‑force or credential‑guessing attack against the SSH service. The server rejected each attempt and closed the connections before any successful login could occur.
- routing_path: semantic | task: Summarization | review_status: in_review

**[OPENSTACK] openstack_v1_semantic_nova_osapi_compute_wsgi_server_0**
- Soru: Looking at these OpenStack log lines, what happened and what does it mean?
- Cevap: The log shows repeated successful API requests to list servers ending in status 200, followed by a compute node recording that instance **b9000564‑fe1a‑409b‑b8cc‑1e88b294cd1d** experienced a "VM Started" lifecycle event and then immediately a "VM Paused" event. This indicates the virtual machine was launched and then paused shortly thereafter.
- routing_path: semantic | task: Summarization | review_status: in_review

**[WINDOWS] windows_v1_semantic_csi_0**
- Soru: Looking at these Windows log lines, what happened and what does it mean?
- Cevap: The excerpt shows that at 2016‑09‑28 04:30:31 the TrustedInstaller service was initialized and started successfully by the CBS subsystem, completing its initialization phase. This indicates a normal Windows component servicing process beginning without errors.
- routing_path: semantic | task: Summarization | review_status: in_review

**[ZOOKEEPER] zookeeper_v1_semantic_fastleaderelection_0**
- Soru: Looking at these Zookeeper log lines, what happened and what does it mean?
- Cevap: The log shows repeated connection failures and restarts of a quorum election process on server myid=1, indicating it is losing connections to peers and attempting to re-establish them. The warnings about SendWorker leaving threads and RecvWorker reporting broken connections suggest the server is experiencing instability in its communication channels with other Zookeeper ensemble members.
- routing_path: semantic | task: Summarization | review_status: in_review

## Easy (deterministik: count / presence / lookup)

**[APACHE] apache_v1_count_notice_0**
- Soru: How many log lines contain '[notice]'?
- Cevap: 1405
- routing_path: sql | task: Aggregation | review_status: verified

**[APACHE] apache_v1_presence_mod_jk_0**
- Soru: Does the log contain any line with 'mod_jk'?
- Cevap: Yes
- routing_path: sql | task: Aggregation | review_status: verified

**[APACHE] apache_v1_lookup_child_workerenv_in_error_state_first_0**
- Soru: Show the first line that reports 'child workerEnv in error state'.
- Cevap: [Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6
- routing_path: keyword | task: Lookup | review_status: verified

**[LINUX] linux_v1_count_authentication_failure_0**
- Soru: How many log lines contain 'authentication failure'?
- Cevap: 490
- routing_path: sql | task: Aggregation | review_status: verified

**[LINUX] linux_v1_presence_invalid_user_0**
- Soru: Does the log contain any line with 'Invalid user'?
- Cevap: No
- routing_path: sql | task: Aggregation | review_status: verified

**[LINUX] linux_v1_lookup_authentication_failure_first_0**
- Soru: Show the first line that reports 'authentication failure'.
- Cevap: Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.188.2.4 
- routing_path: keyword | task: Lookup | review_status: verified

**[MAC] mac_v1_count_kernel_0**
- Soru: How many log lines contain 'kernel'?
- Cevap: 775
- routing_path: sql | task: Aggregation | review_status: verified

**[MAC] mac_v1_presence_thermal_pressure_0**
- Soru: Does the log contain any line with 'Thermal pressure'?
- Cevap: Yes
- routing_path: sql | task: Aggregation | review_status: verified

**[MAC] mac_v1_lookup_iothunderboltswitch_first_0**
- Soru: Show the first line that reports 'IOThunderboltSwitch'.
- Cevap: Jul  1 09:00:55 calvisitor-10-105-160-95 kernel[0]: IOThunderboltSwitch<0>(0x0)::listenerCallback - Thunderbolt HPD packet for route = 0x0 port = 11 unplug = 0
- routing_path: keyword | task: Lookup | review_status: verified

**[WINDOWS] windows_v1_count_cbs_0**
- Soru: How many log lines contain 'CBS'?
- Cevap: 1973
- routing_path: sql | task: Aggregation | review_status: verified

**[WINDOWS] windows_v1_presence_error_0**
- Soru: Does the log contain any line with 'Error'?
- Cevap: Yes
- routing_path: sql | task: Aggregation | review_status: verified

**[WINDOWS] windows_v1_lookup_csi_first_0**
- Soru: Show the first line that reports 'CSI'.
- Cevap: 2016-09-28 04:30:31, Info                  CSI    00000001@2016/9/27:20:30:31.455 WcpInitialize (wcp.dll version 0.0.0.6) called (stack @0x7fed806eb5d @0x7fef9fb9b6d @0x7fef9f8358f @0xff83e97c @0xff83d799 @0xff83db2f)
- routing_path: keyword | task: Lookup | review_status: verified
