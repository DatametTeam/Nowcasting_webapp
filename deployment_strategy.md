# Multi-User Deployment Strategy

**Date**: 2025-11-22
**Status**: Design Phase
**Target Users**: 5 concurrent users (small research team)

---

## 📋 Problem Statement

When deploying the nowcasting webapp for multiple users, several coordination issues arise:

### Identified Issues

1. **Real-time prediction conflicts**
   - Multiple users triggering predictions for the same timestamp every 5 minutes
   - Wasteful duplicate computation
   - Potential file write conflicts

2. **Concurrent file access**
   - User A deleting/recomputing predictions while User B visualizes them
   - Race conditions on file reads/writes
   - GIF generation conflicts

3. **PBS queue flooding**
   - N users × M models = many jobs every 5 minutes
   - Could overwhelm HPC queue

4. **Storage explosion**
   - Multiple users creating redundant GIFs for same timestamps
   - Disk space waste

5. **Resource contention**
   - Multiple real-time background threads monitoring same files
   - Unnecessary CPU/memory usage

6. **No user isolation**
   - All predictions/GIFs are shared
   - One user's actions affect others

---

## ✅ Solution Overview

### Core Strategy: PBS Queue Coordination + Shared State

**Principle**: Use the PBS job queue as a natural coordination mechanism, enhanced with a lightweight shared state file for better UX.

### Key Components

1. **Job Deduplication via PBS Queue**
   - Before submitting: check if job already exists in queue
   - If exists: monitor existing job instead of submitting duplicate
   - Job names encode model + timestamp for easy identification

2. **Shared State JSON File**
   - Tracks active jobs with metadata (who, when, status)
   - Enables better UI messages ("User X is generating predictions")
   - Faster than querying PBS every time
   - Automatic cleanup of old entries

3. **Deletion Protection**
   - Check for active jobs before allowing deletion
   - Warn user if others are generating those predictions
   - Allow override with explicit confirmation

4. **Timestamp Normalization**
   - Round real-time timestamps to nearest 5-minute interval
   - Ensures all users check for same job name
   - Critical for preventing race conditions

---

## 🏗️ Technical Design

### Job Naming Convention

**Requirements:**
- Short (PBS has ~15 char limit on some systems)
- Unique per model + timestamp/range
- Easy to parse

**Format:**

```
Real-time:    nwc_rt_{model}_{YYMMDDHHMM}
              Example: nwc_rt_ConvLSTM_2511221400

Nowcasting:   nwc_nr_{model}_{start}_{end}
              Example: nwc_nr_ConvLSTM_2511221400_1500

              Or with hash for long ranges:
              nwc_nr_{hash8}
              Example: nwc_nr_a3f5c2e1
```

### Timestamp Normalization

**Purpose**: Ensure all users look for same job name at any given time.

**Implementation:**
```python
def normalize_timestamp(dt):
    """Round to nearest 5-minute interval"""
    minute = (dt.minute // 5) * 5
    return dt.replace(minute=minute, second=0, microsecond=0)

# User A at 14:03:42 → 14:05:00
# User B at 14:04:17 → 14:05:00
# Both check for: nwc_rt_ConvLSTM_2511221405
```

### Shared State File Structure

**Location**: `/davinci-1/work/protezionecivile/nwc_webapp/shared_state.json`

**Format:**
```json
{
  "realtime": {
    "ConvLSTM_2511221400": {
      "job_id": "12345.dvgpu006",
      "user": "matte",
      "submitted_at": "2025-11-22T14:00:03",
      "status": "queued",
      "last_updated": "2025-11-22T14:00:03"
    }
  },
  "nowcasting": {
    "ConvLSTM_2511221400_1500": {
      "job_id": "12346.dvgpu006",
      "user": "paolo",
      "submitted_at": "2025-11-22T14:05:10",
      "start_time": "2025-11-22T14:00:00",
      "end_time": "2025-11-22T15:00:00",
      "status": "running",
      "last_updated": "2025-11-22T14:10:15"
    }
  }
}
```

**Job Status States:**
- `queued`: Job submitted, waiting in PBS queue
- `running`: Job executing on compute node
- `completed`: Job finished successfully
- `failed`: Job encountered error

### Concurrency Handling

**Method**: File locking (`fcntl`) + Atomic writes

**Why this works:**
- `fcntl.flock()`: OS-level exclusive lock
  - Process blocks until lock available
  - Automatic release on file close (even on crash)
  - Works on shared filesystems (NFS/Lustre)

- Atomic writes (temp file + `os.replace()`):
  - Write to temp file first
  - Atomic rename (POSIX guarantee)
  - Readers never see partial data

**Write Pattern:**
1. Open file with exclusive lock
2. Read current state
3. Modify in memory
4. Write to temp file
5. Atomic replace
6. Release lock

### Race Condition Mitigation

**Scenario**: Two users check queue simultaneously before either job appears.

**Solution**: Double-check with small random delay
```python
# First check
if job_exists(): return "existing"

# Random delay (0-2 seconds)
time.sleep(random.uniform(0, 2))

# Second check
if job_exists(): return "existing"

# Submit
submit_job()
```

**For 5 users**: Collision probability is negligible.

### Cleanup Strategy

**Old entries removed when:**
- Age > 24 hours
- Job no longer in PBS queue (completed/failed)

**Cleanup triggers:**
- Lazy: On every write operation
- Periodic: Optional background task (not required)

**Benefits:**
- File stays small
- No stale data
- No separate cleanup process needed

---

## 🔧 Implementation Components

### 1. SharedJobState Class

**File**: `src/nwc_webapp/services/shared_state.py`

**Key Methods:**
- `add_job(job_type, job_key, job_info)` - Add/update job entry
- `get_job(job_type, job_key)` - Retrieve job info
- `get_all_jobs(job_type)` - Get all jobs of type
- `update_job_status(job_type, job_key, status)` - Update status
- `remove_job(job_type, job_key)` - Remove job entry
- `check_active_jobs_for_range(model, start, end)` - Deletion protection
- `_locked_read_write()` - Context manager for safe access
- `_cleanup_old_entries(state, max_age_hours)` - Remove old jobs

**Features:**
- Thread-safe file locking
- Atomic writes
- Automatic cleanup
- Error recovery (corrupted JSON)

### 2. PBS Job Submission Integration

**Files to modify:**
- `src/nwc_webapp/services/pbs.py`
- `src/nwc_webapp/page_modules/real_time.py`
- `src/nwc_webapp/page_modules/nowcasting.py`

**Flow:**
```python
def submit_job_with_dedup(model, timestamp):
    job_key = generate_job_key(model, timestamp)

    # 1. Check shared state (fast)
    existing = state.get_job("realtime", job_key)
    if existing and existing["status"] in ["queued", "running"]:
        return "existing", existing["job_id"]

    # 2. Verify with PBS queue (authoritative)
    if job_exists_in_pbs_queue(job_key):
        # Update state to sync with reality
        state.add_job("realtime", job_key, {...})
        return "existing", None

    # 3. Small delay + recheck (race condition mitigation)
    time.sleep(random.uniform(0, 2))
    if job_exists_in_pbs_queue(job_key):
        return "existing", None

    # 4. Submit new job
    job_id = submit_pbs_job(..., job_name=job_key)

    # 5. Record in shared state
    state.add_job("realtime", job_key, {
        "job_id": job_id,
        "user": getpass.getuser(),
        "submitted_at": datetime.now().isoformat(),
        "status": "queued"
    })

    return "submitted", job_id
```

### 3. Deletion Protection UI

**Files to modify:**
- `src/nwc_webapp/page_modules/nowcasting.py`

**Flow:**
```python
def handle_delete_predictions():
    # Check for active jobs in range
    active_jobs = state.check_active_jobs_for_range(model, start, end)

    if active_jobs:
        st.warning(f"⚠️ {len(active_jobs)} job(s) generating predictions:")
        for job in active_jobs:
            st.write(f"- Job {job['job_id']} by {job['user']}")

        if st.button("⚠️ Delete Anyway"):
            delete_predictions()
            st.info("Deleted (others may be affected)")
    else:
        if st.button("Delete Predictions"):
            delete_predictions()
            st.success("Deleted successfully")
```

### 4. Helper Functions

**File**: `src/nwc_webapp/services/pbs_utils.py` (new file)

```python
def normalize_timestamp(dt: datetime) -> datetime:
    """Round to nearest 5-minute interval"""

def generate_realtime_job_key(model: str, timestamp: datetime) -> str:
    """Generate standardized job name for real-time prediction"""

def generate_nowcasting_job_key(model: str, start: datetime, end: datetime) -> str:
    """Generate standardized job name for nowcasting"""

def job_exists_in_pbs_queue(job_name: str) -> bool:
    """Check if job exists in PBS queue (running or queued)"""

def parse_qstat_output(output: str) -> List[Dict[str, Any]]:
    """Parse qstat output into structured data"""
```

---

## ✅ TODO List

### Phase 1: Core Infrastructure

- [ ] **Create SharedJobState class**
  - [ ] File: `src/nwc_webapp/services/shared_state.py`
  - [ ] Implement `fcntl` file locking
  - [ ] Implement atomic writes
  - [ ] Add all CRUD methods
  - [ ] Add cleanup logic
  - [ ] Add error handling
  - [ ] Write unit tests

- [ ] **Create PBS utilities**
  - [ ] File: `src/nwc_webapp/services/pbs_utils.py`
  - [ ] `normalize_timestamp()` function
  - [ ] `generate_realtime_job_key()` function
  - [ ] `generate_nowcasting_job_key()` function
  - [ ] `job_exists_in_pbs_queue()` function
  - [ ] `parse_qstat_output()` function
  - [ ] Write unit tests

- [ ] **Update config**
  - [ ] Add `shared_state_file` path to `cfg.yaml`
  - [ ] Add to both HPC and local paths
  - [ ] Document in CLAUDE.md

### Phase 2: Real-Time Prediction Integration

- [ ] **Modify `src/nwc_webapp/services/pbs.py`**
  - [ ] Add `job_name` parameter to `submit_realtime_pbs_job()`
  - [ ] Integrate job deduplication logic
  - [ ] Add shared state recording
  - [ ] Add job status updates during monitoring

- [ ] **Modify `src/nwc_webapp/page_modules/real_time.py`**
  - [ ] Import SharedJobState
  - [ ] Update "Start Prediction" button logic
  - [ ] Check shared state before submitting
  - [ ] Display existing job info if found
  - [ ] Show who submitted the job
  - [ ] Update UI messages for clarity

- [ ] **Add timestamp normalization**
  - [ ] Apply to all real-time prediction triggers
  - [ ] Ensure consistent job key generation
  - [ ] Update logging to show normalized time

### Phase 3: Nowcasting Tab Integration

- [ ] **Modify `src/nwc_webapp/page_modules/nowcasting.py`**
  - [ ] Import SharedJobState
  - [ ] Update job submission logic
  - [ ] Check shared state before submitting
  - [ ] Display existing job info if found
  - [ ] Show who submitted the job

- [ ] **Modify `src/nwc_webapp/page_modules/nowcasting_utils.py`**
  - [ ] Add `check_active_jobs_for_predictions()` function
  - [ ] Integrate with deletion workflow

- [ ] **Add deletion protection UI**
  - [ ] Check for active jobs before deletion
  - [ ] Display warning with job details
  - [ ] Add "Delete Anyway" confirmation button
  - [ ] Log deletion actions

### Phase 4: Job Monitoring Improvements

- [ ] **Enhance job monitoring**
  - [ ] Update job status in shared state during monitoring
  - [ ] Handle job completion (update state to "completed")
  - [ ] Handle job failure (update state to "failed")
  - [ ] Clean up completed jobs from state

- [ ] **Add job status synchronization**
  - [ ] Periodic sync between PBS queue and shared state
  - [ ] Detect orphaned state entries (job not in queue)
  - [ ] Detect missing state entries (job in queue but not state)

### Phase 5: UI Enhancements

- [ ] **Add shared state viewer (optional)**
  - [ ] Sidebar section showing active jobs
  - [ ] Display: model, user, status, time
  - [ ] Collapsible/expandable
  - [ ] Auto-refresh

- [ ] **Improve status messages**
  - [ ] "Job already submitted by {user} at {time}"
  - [ ] "Monitoring existing job: {job_id}"
  - [ ] "Job completed by {user}"
  - [ ] "{N} users are generating predictions in this range"

- [ ] **Add user identification**
  - [ ] Display current user in sidebar
  - [ ] Show in job submission messages
  - [ ] Log in shared state

### Phase 6: Testing & Validation

- [ ] **Unit tests**
  - [ ] SharedJobState class tests
  - [ ] PBS utilities tests
  - [ ] Concurrency tests (multiple processes)
  - [ ] Cleanup logic tests

- [ ] **Integration tests**
  - [ ] Simulated multi-user scenarios
  - [ ] Real-time deduplication test
  - [ ] Nowcasting deduplication test
  - [ ] Deletion protection test
  - [ ] Race condition tests

- [ ] **Manual testing**
  - [ ] Test with 2+ concurrent users
  - [ ] Real-time prediction conflicts
  - [ ] Nowcasting job overlap
  - [ ] Deletion while job running
  - [ ] Job monitoring accuracy
  - [ ] State file corruption recovery

### Phase 7: Documentation & Deployment

- [ ] **Update documentation**
  - [ ] Update CLAUDE.md with multi-user notes
  - [ ] Document shared state file format
  - [ ] Add troubleshooting guide
  - [ ] Add multi-user usage instructions

- [ ] **Create deployment guide**
  - [ ] Initial setup steps
  - [ ] Shared state file location
  - [ ] File permissions setup
  - [ ] User onboarding instructions

- [ ] **Add logging**
  - [ ] Log all job submissions
  - [ ] Log deduplication events
  - [ ] Log deletion actions
  - [ ] Log state file operations

- [ ] **Monitoring setup**
  - [ ] Log file location for multi-user events
  - [ ] Optional: Dashboard for active jobs
  - [ ] Optional: Metrics collection

---

## 🧪 Testing Strategy

### Unit Tests

```python
# tests/test_shared_state.py
def test_add_job():
    state = SharedJobState(temp_file)
    state.add_job("realtime", "test_key", {...})
    job = state.get_job("realtime", "test_key")
    assert job is not None

def test_concurrent_writes():
    # Test with multiprocessing
    processes = [Process(target=write_job) for _ in range(5)]
    # Verify no data loss
```

### Integration Tests

```python
# tests/test_deduplication.py
def test_realtime_deduplication():
    # Simulate 2 users submitting same prediction
    # Verify only 1 job submitted

def test_deletion_protection():
    # Start job
    # Attempt deletion
    # Verify warning shown
```

### Manual Test Scenarios

1. **Scenario 1: Simultaneous Real-Time**
   - 2 users open real-time tab
   - Both click "Start Prediction" within 1 second
   - **Expected**: Only 1 job submitted, both monitor same job

2. **Scenario 2: Nowcasting Overlap**
   - User A generates 14:00-15:00
   - User B generates 14:30-15:30
   - User B tries to delete 14:30-15:00
   - **Expected**: Warning shown about User A's job

3. **Scenario 3: State Recovery**
   - Corrupt shared_state.json
   - Submit new job
   - **Expected**: State file reinitialized, job recorded

---

## 🔮 Future Considerations

### Short Term (After Initial Deployment)

- [ ] **Job queue viewer**: Web UI to see all active jobs
- [ ] **User notifications**: Alert when someone deletes your data
- [ ] **Job priority**: Admin users get priority in queue
- [ ] **Storage quotas**: Limit GIF storage per user

### Medium Term

- [ ] **User workspaces**: Personal directories for experiments
- [ ] **Job scheduling**: Queue predictions for specific times
- [ ] **Metrics dashboard**: Track usage, job counts, wait times
- [ ] **Automated cleanup**: Background task to clean old predictions

### Long Term (If Scaling Beyond 5 Users)

- [ ] **Database backend**: Replace JSON with SQLite/PostgreSQL
- [ ] **Job pooling**: Single job for multiple users' requests
- [ ] **Distributed locking**: If moving beyond single filesystem
- [ ] **User authentication**: Proper login system
- [ ] **Role-based access**: Admin vs. regular user permissions

---

## 📝 Notes & Design Decisions

### Why PBS Queue + JSON (Not Database)?

**Pros:**
- ✅ No new dependencies
- ✅ Simple implementation
- ✅ Fast enough for 5 users
- ✅ Easy to debug (human-readable JSON)
- ✅ Leverages existing PBS infrastructure

**Cons:**
- ⚠️ JSON file can't handle 100+ concurrent users
- ⚠️ No advanced queries
- ⚠️ Manual cleanup needed

**Decision**: JSON is perfect for 5 users, can migrate to DB later if needed.

### Why fcntl (Not Database/Redis)?

**Pros:**
- ✅ No external services
- ✅ Works on shared filesystem
- ✅ Automatic cleanup
- ✅ Simple to understand

**Cons:**
- ⚠️ Linux-specific (but HPC is Linux)
- ⚠️ NFS caveats (but usually works)

**Decision**: fcntl is the right tool for this use case.

### Why Full Rewrite (Not Append-Only)?

**Pros:**
- ✅ File stays small
- ✅ Easier to clean up old entries
- ✅ Simpler logic

**Cons:**
- ⚠️ Slightly more I/O

**Decision**: File is tiny (5 users × 10 jobs = <10KB), full rewrite is fine.

### Acceptable Risks (Team of 5)

Given the small team size, these edge cases are acceptable:
- Someone manually deletes predictions without checking webapp
- Two users simultaneously click "Delete Anyway" (rare)
- Job fails but state not updated (manual cleanup)
- State file corruption (auto-recovery implemented)

---

## 📚 References

- [fcntl documentation](https://docs.python.org/3/library/fcntl.html)
- [POSIX file locking](https://en.wikipedia.org/wiki/File_locking#In_Unix-like_systems)
- [Atomic file writes](https://danluu.com/file-consistency/)
- PBS Job Management: `qstat`, `qdel`, `qsub` commands

---

**Last Updated**: 2025-11-22
**Author**: Deployment strategy discussion with team
**Status**: Ready for implementation