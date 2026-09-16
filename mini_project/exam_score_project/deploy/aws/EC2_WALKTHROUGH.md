# Manual EC2 Walkthrough (do this once, by hand, before trusting the script)

`ec2-user-data.sh` in this folder automates everything below in one shot. **Do it manually
first anyway.** The whole point of this walkthrough is that you should be able to explain
every line of that script afterwards — running a script you don't understand isn't a skill,
it's a leap of faith. Budget ~20 minutes the first time.

You need: an AWS account (the Free Tier covers a `t2.micro`/`t3.micro` for a year), and the
image already pushed to Docker Hub (Step 2 of the main guide — `DOCKER_STREAMLIT_AWS_GUIDE.md`
at the project root).

---

## Step 1 — Launch the instance

1. AWS Console → **EC2** → **Launch instance**.
2. **Name**: `exam-score-demo` (anything — just so you recognize it later).
3. **AMI**: *Ubuntu Server 22.04 LTS* (the "Free tier eligible" one at the top of the list).
4. **Instance type**: `t2.micro` (or `t3.micro`) — Free Tier eligible.
5. **Key pair**: create a new one (e.g. `exam-score-key`), download the `.pem` file, and keep
   it — it's the only way you'll SSH in, and AWS never lets you download it again.
6. **Network settings → Edit** — this is the step students most often skip and then can't
   reach their own app. Add rules so the security group looks like this:

   | Type | Protocol | Port range | Source | Why |
   |---|---|---|---|---|
   | SSH | TCP | 22 | My IP | so *you* can SSH in — never `0.0.0.0/0` for SSH on a real box |
   | Custom TCP | TCP | 8000 | Anywhere (0.0.0.0/0) | the FastAPI service |
   | Custom TCP | TCP | 8501 | Anywhere (0.0.0.0/0) | the Streamlit UI |

   A **security group** is AWS's firewall for the instance — nothing on those ports is
   reachable from outside until a rule explicitly opens it, no matter what the container
   inside is doing. This is the step that turns "container runs" into "container is actually
   reachable from my laptop."
7. Leave storage at the default 8 GB. Click **Launch instance**.

## Step 2 — Connect

Wait ~1 minute for the instance state to show **Running** and **2/2 status checks passed**,
then copy its **Public IPv4 address** from the instance list.

```bash
chmod 400 exam-score-key.pem
ssh -i exam-score-key.pem ubuntu@<public-ip>
```

(`chmod 400` — SSH refuses to use a key file that other users on your machine could read.)

## Step 3 — Install Docker by hand

This is exactly what `ec2-user-data.sh` does automatically — typing it yourself once is what
makes the script legible afterwards instead of magic.

```bash
sudo apt-get update -y
sudo apt-get install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker

# so you can run `docker` without typing sudo every time (takes effect on your NEXT login)
sudo usermod -aG docker ubuntu
exit
```

SSH back in (so the group membership actually applies), then confirm:

```bash
ssh -i exam-score-key.pem ubuntu@<public-ip>
docker --version
```

## Step 4 — Pull and run the combined image

```bash
docker pull <your-dockerhub-username>/exam-score-app:1.0

docker run -d \
    --name exam-score-app \
    --restart unless-stopped \
    -p 8000:8000 \
    -p 8501:8501 \
    <your-dockerhub-username>/exam-score-app:1.0
```

- `-d` — detached, keeps running after you close the SSH session.
- `--restart unless-stopped` — if the instance reboots (a real event: AWS maintenance, or you
  restarting it), the container comes back up on its own without you SSHing in again.
- Both `-p` flags matter — the API and the UI are two different processes inside the same
  container (see `docker/entrypoint.sh`), and each needs its own port published.

## Step 5 — Verify from your own laptop

Not from inside the SSH session — from your browser, to prove the security group rules
actually work:

- `http://<public-ip>:8000/docs` → FastAPI's Swagger UI
- `http://<public-ip>:8501` → the Streamlit app

If these hang instead of loading, it's almost always Step 1's security group, not Docker —
check that the rules were actually saved.

## Step 6 — Useful commands while it's live

```bash
docker ps                          # is it running?
docker logs -f exam-score-app      # tail both services' combined output
docker restart exam-score-app      # restart without re-pulling
docker stop exam-score-app         # stop (entrypoint.sh's SIGTERM handling stops both processes)
```

## Step 7 — Tear it down (don't leave it running and forget it)

```bash
docker stop exam-score-app && docker rm exam-score-app
```

Then, in the AWS Console: **Instance state → Terminate instance**. A stopped instance still
occupies its EBS volume; a *terminated* one doesn't. Free Tier hours are generous but not
infinite — terminate demo instances you're done with.

---

## Now compare to the automated script

Open `ec2-user-data.sh`. Every command in it is one you just typed by hand above, glued
together to run unattended as **EC2 User Data** — a script AWS runs as `root` the moment the
instance first boots, before you ever SSH in. Paste it into **Launch instance → Advanced
details → User data** and the instance arrives already running the container — useful for
spinning up a fresh demo instance quickly, but only trustworthy to hand to a script *after*
you've verified you understand what it's doing by having done it manually here.

## Common mistakes

- **Forgetting to open 8000/8501 in the security group** — the single most common "it works
  locally but not on EC2" report. Docker isn't the problem; the firewall is.
- **Using the instance's *private* IP instead of its *public* IP** — the private IP only
  works from inside AWS's own network.
- **SSH key permissions too open** (`chmod 644` etc.) — SSH silently refuses the key; error
  message talks about "permissions too open," not about the actual connection.
- **Forgetting `--restart unless-stopped`** — the container doesn't survive a reboot, and the
  demo "mysteriously" stops working the next morning.
- **Leaving the instance running after the demo** — set a reminder to terminate it.
