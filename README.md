# Brain MRI Tumour Segmentation — A 10-Hour Seminar

A hands-on seminar for high school students. Students train a U-Net to find brain tumours in real
MRI scans, and — more importantly — learn why the number their model reports might be a lie.

Everything runs on **free Google Colab**. No installs, no accounts, no cost.

---

## The one idea

> Two expert radiologists, shown the same scan, agree on the tumour's location about **84%** of the
> time (Buda et al., 2019 — 84% Dice, sd 2%).

Every session comes back to this. It reframes "what score is good?" from a homework question into a
real one, and it's the honest introduction to medical AI: the ceiling isn't 100%, and nobody knows
exactly where it is.

## Files

| file | who | what |
|---|---|---|
| `00_INSTRUCTOR_prepare_data.ipynb` | **you, once** | Kaggle download → 70 MB `.npz`, hosting instructions |
| `01_looking_at_the_data.ipynb` | students | Session 1 — dataset report, no ML |
| `02_baseline_and_the_metric_trap.ipynb` | students | Session 2 — thresholding, and the accuracy trap |
| `03_build_a_unet.ipynb` | students | Session 3 — write and train a U-Net |
| `04_improve_it_then_break_it.ipynb` | students | Session 4 — improvements, then the leakage reveal |
| `05_capstone.ipynb` | students | Session 5 — three research tracks |
| `seminar.py` | both | shared helper library, fetched by every notebook |
| `prepare_data.py` | you | CLI version of the conversion, if you prefer a terminal |
| `SOLUTIONS.py` | **you only** | every TODO filled in — do not distribute |

## Dataset

**Kaggle LGG MRI Segmentation** (`mateuszbuda/lgg-mri-segmentation`) — 110 lower-grade glioma
patients from TCIA/TCGA, with FLAIR abnormality masks approved by a board-certified radiologist at
Duke. Roughly 3,900 slices, 256×256, three channels (pre-contrast / FLAIR / post-contrast), stored
as `.tif`. Free, no data use agreement.

Notebook `00` downsamples to 128×128 and packs everything into one `~70 MB` `.npz`.

---

## Setup, in order

1. **Run `00_INSTRUCTOR_prepare_data.ipynb`.** You'll need a Kaggle token — yours, once. Students
   never need one.
2. **Host `lgg_128.npz` and `lgg_meta.csv`.** Hugging Face datasets is the best option: free, fast,
   no auth, no download interstitial. GitHub Releases also works. Avoid Google Drive — its
   large-file confirmation page breaks `urlretrieve`.
3. **Edit `seminar.py`:** set `DATA_URL` and `META_URL` to your hosted links.
4. **Push everything to your GitHub repo**, then update `REPO_RAW` at the top of each student
   notebook to point at it.
5. **Do a full dry run** on real data (see below).

### Fallback that saves the day

If the data URL isn't set or fails, `get_data()` falls back to **synthetic demo data** with a loud
warning. Every notebook runs end to end regardless. This means a student with a broken connection
still participates, and you can rehearse before your mirror exists. Demo numbers are meaningless and
the notebook says so.

---

## Schedule

Five 2-hour sessions. Splits cleanly into 10 one-hour blocks if you prefer.

| session | title | the beat it turns on |
|---|---|---|
| 1 | Look at the data before you touch a model | ~1% of pixels are tumour |
| 2 | Segmentation without deep learning | the all-zeros model scores 99% |
| 3 | Build a U-Net | skip connections; beating the baseline |
| 4 | Make it better, then break it | the split was leaking the whole time |
| 5 | Capstone | your number, your caveat |

### The Session 3→4 trap

**Session 3 deliberately uses `split_by_slice`, which leaks.** This is intentional and Session 4
discloses it explicitly ("I knew it was wrong when I wrote it").

Adjacent slices from one patient are near-identical images. Shuffling slices puts near-copies on
both sides of the split, so the model can score well by memorising. Students discover this by
counting how many patients appear in both sets, then retraining both ways across three seeds.

Two things matter in how you run this:

- **Disclose it clearly.** "I gave you a broken notebook on purpose" lands well. Letting students
  believe they made the mistake themselves does not.
- **Verify the gap appears in your dry run.** The session is built on it. It should exceed the
  seed-to-seed noise. If it doesn't, either tune epochs/val fraction until it's stable, or teach it
  as "sometimes leakage is subtle — which is exactly what makes it dangerous."

---

## Capstone tracks

| | question | risk |
|---|---|---|
| **A** | How many annotated patients do you actually need? | safe; the curve always shows something |
| **B** | Is Dice biased against small tumours? | robust; the effect is strong and real |
| **C** | Can tumour shape predict genomic subtype? | may find nothing — brief them first |

Track C reproduces the source paper's actual finding using students' own predicted masks. It's the
most exciting pitch and the one most likely to produce a null result, since 110 patients across
several clusters is very little power. It includes a **multiple-comparisons check** that shows how
many "significant" hits you'd expect by chance — which is arguably the most valuable thing in the
whole seminar. Frame a null result as a real result up front, not as consolation afterwards.

---

## Before you teach: dry-run checklist

Run 01→05 on the real data and **write your numbers down.** Students will ask "is mine right?"

- [ ] Session 2 — best Dice from thresholding
- [ ] Session 3 — U-Net Dice on the **slice** split
- [ ] Session 4 — U-Net Dice on the **patient** split, and the gap
- [ ] Session 4 — wall-clock time for 12 epochs on a T4
- [ ] Session 5 — one full run of each track you plan to offer

For reference, the original paper reports **83.6% mean Dice** with 22-fold cross-validation. Students
training 12–25 epochs on a subset will land below that, which is fine and worth saying out loud.

## Practical notes

- **GPU:** Runtime → Change runtime type → T4. Sessions 3–5 assert this and fail loudly otherwise.
- **Colab wipes files on disconnect.** Tell students to download `unet_session3.pt` at the end of
  Session 3, or mount Drive.
- **Free Colab disconnects on idle.** Sessions 4 and 5 have multi-run cells; warn students not to
  close the tab.
- **Timing:** 12 epochs at 128×128 on a T4 is a few minutes. Session 4's six-run comparison is the
  longest cell in the seminar — start it, then talk while it runs.
- **Pin versions and re-test the week before.** Colab's preinstalled packages shift.
- `seminar.py` handles the skimage 0.26 `regionprops` rename, so it works on old and new Colab images.

## Ethics — budget 20 minutes, don't lecture

Concrete questions that work better than a slide:

- One radiologist drew these masks. Another would draw them differently. Which one is "correct"?
- All 110 patients came from five US institutions. What happens to this model in a hospital in Morocco?
- Your model scores 0.80. Would you tell a patient that? What *would* you tell them?
- What should happen when the model is confidently wrong?

## Trailer pitch (first Monday)

Six slides, 60–90 seconds:

1. A raw FLAIR slice, unannotated. *"Somewhere in this image is a brain tumour."*
2. The mask overlays.
3. *"Two expert radiologists only agree on where it is 84% of the time."*
4. Your trained model's prediction animating on.
5. The three capstone questions.
6. **"In 10 hours, you'll build an AI that finds brain tumours — and figure out how to know if it's
   any good."**

---

## Credit

Dataset and pretrained reference model: Mateusz Buda, Ashirbani Saha, Maciej A. Mazurowski,
*Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted
by a deep learning algorithm*, Computers in Biology and Medicine, 2019.
Images from The Cancer Imaging Archive (TCIA) / The Cancer Genome Atlas (TCGA).

These materials are for education. Nothing here is a clinical tool.
