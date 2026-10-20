//! Same-host POSIX peer handles, opted into with `KERN_LOCAL_IPC_DIR`.
//!
//! A fabric handle needs an IMEX channel. A host without one (an HGX B300,
//! say) can still share its VMM allocations between processes as POSIX file
//! descriptors, but a descriptor is only a number inside the process that
//! holds it, so the 64 bytes a [`PeerHandle::Posix`](crate::PeerHandle) carries
//! on the wire are a locator, not the handle: a magic, a digest of this
//! boot's id, the exporting process's broker (its pid and a random nonce,
//! which name its socket in the directory) and the key of one export. The
//! importer connects to that socket, the broker passes the descriptor over
//! with `SCM_RIGHTS`, and `cuMemImportFromShareableHandle` turns it into an
//! allocation handle the importer owns. Nothing CUDA hands out crosses the
//! caller's own transport.
//!
//! Every process of a group sets the same directory. The runtime creates it
//! 0700 and refuses one that is not this uid's alone, and each end of a
//! connection checks the other's uid. A locator from another boot is refused
//! before anything connects, so a handle that crossed hosts fails loudly. An
//! export lives as long as the allocation that made it: dropping the
//! allocation closes the broker's descriptor, and mappings already imported
//! stay valid. A broker's socket file outlives its process; the nonce keeps
//! the next process's name apart.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cudarc::driver::sys;
use sha2::{Digest, Sha256};

use crate::error::{bail, cuda_check, Result};

/// The first 16 bytes of a POSIX locator; a fabric handle never starts so.
pub(crate) const MAGIC: &[u8; 16] = b"KERN_LOCAL_FD_v1";

static BROKER: Mutex<Option<Arc<Broker>>> = Mutex::new(None);

/// This process's descriptors on offer, served by one listener thread.
struct Broker {
    nonce: [u8; 16],
    host: [u8; 16],
    next: AtomicU64,
    fds: Arc<Mutex<BTreeMap<u64, Arc<OwnedFd>>>>,
}

/// One allocation's registration with the broker, owned by the allocation:
/// dropping it closes the broker's descriptor; handles already imported
/// stay valid.
pub(crate) struct Export {
    broker: Arc<Broker>,
    key: u64,
    pub(crate) wire: [u8; 64],
}

impl Drop for Export {
    fn drop(&mut self) {
        self.broker.fds.lock().unwrap().remove(&self.key);
    }
}

/// Whether this process creates its VMM allocations with POSIX handles.
pub(crate) fn enabled() -> bool {
    std::env::var_os("KERN_LOCAL_IPC_DIR").is_some()
}

fn host_id() -> Result<[u8; 16]> {
    let boot = std::fs::read("/proc/sys/kernel/random/boot_id")?;
    Ok(Sha256::digest(boot)[..16].try_into().unwrap())
}

fn directory() -> Result<PathBuf> {
    let Some(path) = std::env::var_os("KERN_LOCAL_IPC_DIR").map(PathBuf::from) else {
        bail!(Api, "a POSIX peer handle needs KERN_LOCAL_IPC_DIR set on this host");
    };
    if !path.is_absolute() {
        bail!(Api, "KERN_LOCAL_IPC_DIR must be absolute, got {}", path.display());
    }
    match std::fs::DirBuilder::new().mode(0o700).create(&path) {
        Ok(()) => (),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(e) => return Err(e.into()),
    }
    let m = std::fs::symlink_metadata(&path)?;
    if !m.is_dir() || m.uid() != unsafe { libc::geteuid() } || m.mode() & 0o077 != 0 {
        bail!(Api, "KERN_LOCAL_IPC_DIR {}: want a directory of this uid, mode 0700", path.display());
    }
    Ok(path)
}

fn socket_path(dir: PathBuf, pid: u32, nonce: &[u8]) -> Result<PathBuf> {
    let p = dir.join(format!("{pid}-{}.sock", hex::encode(nonce)));
    // sun_path holds 108 bytes on Linux; keep a margin.
    if p.as_os_str().len() >= 104 {
        bail!(Api, "local IPC socket path is too long: {}", p.display());
    }
    Ok(p)
}

/// Refuses a peer of another uid and bounds every read and write after.
fn check_uid(s: &UnixStream) -> Result<()> {
    let mut peer: libc::ucred = unsafe { std::mem::zeroed() };
    let mut n = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    if unsafe {
        libc::getsockopt(
            s.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            &mut peer as *mut _ as *mut libc::c_void,
            &mut n,
        )
    } != 0
    {
        return Err(std::io::Error::last_os_error().into());
    }
    if peer.uid != unsafe { libc::geteuid() } {
        bail!(Api, "local IPC peer uid {} is not this process's", peer.uid);
    }
    s.set_read_timeout(Some(Duration::from_secs(5)))?;
    s.set_write_timeout(Some(Duration::from_secs(5)))?;
    Ok(())
}

fn send_fd(s: &UnixStream, fd: RawFd) -> Result<()> {
    let mut byte = [1u8];
    let mut iov = libc::iovec { iov_base: byte.as_mut_ptr().cast(), iov_len: 1 };
    // usize storage gives cmsghdr its required alignment.
    let mut control = [0usize; 8];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    msg.msg_controllen = unsafe { libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as u32) } as usize;
    unsafe {
        let c = libc::CMSG_FIRSTHDR(&msg);
        (*c).cmsg_level = libc::SOL_SOCKET;
        (*c).cmsg_type = libc::SCM_RIGHTS;
        (*c).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as u32) as usize;
        std::ptr::write_unaligned(libc::CMSG_DATA(c).cast::<RawFd>(), fd);
        if libc::sendmsg(s.as_raw_fd(), &msg, libc::MSG_NOSIGNAL) != 1 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}

fn recv_fd(s: &UnixStream) -> Result<OwnedFd> {
    let mut byte = [0u8];
    let mut iov = libc::iovec { iov_base: byte.as_mut_ptr().cast(), iov_len: 1 };
    let mut control = [0usize; 8];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    msg.msg_controllen = std::mem::size_of_val(&control);
    let n = unsafe { libc::recvmsg(s.as_raw_fd(), &mut msg, libc::MSG_CMSG_CLOEXEC) };
    if n < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // Take ownership of every descriptor that arrived, so extras close.
    let mut fds = Vec::new();
    unsafe {
        let mut c = libc::CMSG_FIRSTHDR(&msg);
        while !c.is_null() {
            if (*c).cmsg_level == libc::SOL_SOCKET && (*c).cmsg_type == libc::SCM_RIGHTS {
                let len = (*c).cmsg_len.saturating_sub(libc::CMSG_LEN(0) as usize) / std::mem::size_of::<RawFd>();
                for i in 0..len {
                    fds.push(OwnedFd::from_raw_fd(std::ptr::read_unaligned(libc::CMSG_DATA(c).cast::<RawFd>().add(i))));
                }
            }
            c = libc::CMSG_NXTHDR(&msg, c);
        }
    }
    if n != 1 || byte[0] != 1 || msg.msg_flags & libc::MSG_CTRUNC != 0 || fds.len() != 1 {
        bail!(Api, "local IPC broker did not return exactly one descriptor (gone, or the export was dropped)");
    }
    Ok(fds.pop().unwrap())
}

/// This process's broker, started on first export: a socket in the
/// directory and a thread that answers each connection's key with its
/// descriptor.
fn broker() -> Result<Arc<Broker>> {
    let mut shared = BROKER.lock().unwrap();
    if let Some(b) = &*shared {
        return Ok(b.clone());
    }
    let mut nonce = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut nonce)?;
    let socket = socket_path(directory()?, std::process::id(), &nonce)?;
    let listener = UnixListener::bind(&socket)?;
    std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600))?;
    let fds = Arc::new(Mutex::new(BTreeMap::<u64, Arc<OwnedFd>>::new()));
    let registry = fds.clone();
    std::thread::Builder::new().name("kern-local-ipc".into()).spawn(move || {
        for conn in listener.incoming() {
            let served: Result<()> = (|| {
                let mut s = conn?;
                check_uid(&s)?;
                let mut key = [0u8; 8];
                s.read_exact(&mut key)?;
                let fd = registry.lock().unwrap().get(&u64::from_le_bytes(key)).cloned();
                if let Some(fd) = fd {
                    send_fd(&s, fd.as_raw_fd())?;
                }
                Ok(())
            })();
            if let Err(e) = served {
                tracing::warn!("local IPC broker: {e}");
            }
        }
    })?;
    let b = Arc::new(Broker { nonce, host: host_id()?, next: AtomicU64::new(1), fds });
    *shared = Some(b.clone());
    Ok(b)
}

/// Registers an allocation created with a POSIX handle type; its locator
/// is the returned export's `wire`.
pub(crate) fn export(handle: sys::CUmemGenericAllocationHandle) -> Result<Export> {
    let b = broker()?;
    let mut raw: RawFd = -1;
    cuda_check(
        unsafe {
            sys::cuMemExportToShareableHandle(
                (&mut raw as *mut RawFd).cast(),
                handle,
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            )
        },
        "cuMemExportToShareableHandle(POSIX)",
    )?;
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    // The broker's descriptor must not leak into a child the process execs.
    if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let key = b.next.fetch_add(1, Ordering::Relaxed);
    let mut wire = [0u8; 64];
    wire[..16].copy_from_slice(MAGIC);
    wire[16..32].copy_from_slice(&b.host);
    wire[32..48].copy_from_slice(&b.nonce);
    wire[48..52].copy_from_slice(&std::process::id().to_le_bytes());
    wire[52..60].copy_from_slice(&key.to_le_bytes());
    b.fds.lock().unwrap().insert(key, Arc::new(fd));
    Ok(Export { broker: b, key, wire })
}

/// The allocation handle behind a locator, owned by the caller.
pub(crate) fn import(wire: &[u8; 64]) -> Result<sys::CUmemGenericAllocationHandle> {
    if &wire[..16] != MAGIC || wire[16..32] != host_id()? || wire[60..] != [0; 4] {
        bail!(Api, "a POSIX peer handle not from this host, or in a format this build does not read");
    }
    let pid = u32::from_le_bytes(wire[48..52].try_into().unwrap());
    let mut s = UnixStream::connect(socket_path(directory()?, pid, &wire[32..48])?)?;
    check_uid(&s)?;
    s.write_all(&wire[52..60])?;
    let fd = recv_fd(&s)?;
    let mut handle = 0;
    cuda_check(
        unsafe {
            sys::cuMemImportFromShareableHandle(
                &mut handle,
                fd.as_raw_fd() as usize as *mut _,
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            )
        },
        "cuMemImportFromShareableHandle(POSIX)",
    )?;
    // `fd` closes here either way; the imported handle does not need it.
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scm_rights_duplicates_the_descriptor_and_the_copy_closes_on_drop() {
        let (a, b) = UnixStream::pair().unwrap();
        let file = std::fs::File::open("/proc/sys/kernel/random/boot_id").unwrap();
        send_fd(&a, file.as_raw_fd()).unwrap();
        let fd = recv_fd(&b).unwrap();
        let raw = fd.as_raw_fd();
        let cloexec = unsafe { libc::fcntl(raw, libc::F_GETFD) } & libc::FD_CLOEXEC != 0;
        drop(fd);
        assert_eq!((raw != file.as_raw_fd(), cloexec, unsafe { libc::fcntl(raw, libc::F_GETFD) }), (true, true, -1));
    }

    #[test]
    fn a_locator_from_another_host_is_refused_before_connecting() {
        let mut wire = [0u8; 64];
        wire[..16].copy_from_slice(MAGIC);
        assert!(import(&wire).unwrap_err().to_string().contains("not from this host"));
    }
}
