/* Testing console -- client side. Vanilla, no build step, no framework.
 *
 * Everything here is a thin shell over the JSON routes in app.py. Nothing is
 * computed in the browser that the server could be asked for, because the
 * server is the thing whose answers we are trying to measure.
 */
(function () {
  "use strict";

  var WFF = window.WFF || {};
  var qs = function (sel, root) { return (root || document).querySelector(sel); };
  var qsa = function (sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  };

  /* -- who is testing ---------------------------------------------------
   * Asked once, then remembered. It is stamped on every run and every
   * judgement, so "who tested what" is always answerable -- which was the
   * point of asking at all.
   */
  var STORE_KEY = "wff-reviewer";
  var reviewerInput = qs("#reviewer");
  var idask = qs("#idask");

  function reviewer() {
    return (reviewerInput && reviewerInput.value.trim()) || "unknown";
  }

  function paintIdentity(name) {
    if (reviewerInput) reviewerInput.value = name;
    var runField = qs("#run-reviewer");
    if (runField) runField.value = name;
    var nameEl = qs("#id-name");
    var avatarEl = qs("#id-avatar");
    if (nameEl) nameEl.textContent = name || "nobody";
    if (avatarEl) {
      avatarEl.textContent = (name || "?").charAt(0).toUpperCase();
      avatarEl.className = "av" + (name.toLowerCase() === "devesh" ? " d" : "");
    }
  }

  function setIdentity(name) {
    name = (name || "").trim().slice(0, 40);
    if (!name) return;
    localStorage.setItem(STORE_KEY, name);
    paintIdentity(name);
    if (idask) idask.classList.remove("open");
  }

  function askIdentity() {
    if (idask) idask.classList.add("open");
  }

  var savedName = localStorage.getItem(STORE_KEY) || "";
  if (savedName) {
    paintIdentity(savedName);
  } else {
    paintIdentity("");
    askIdentity();
  }

  if (idask) {
    qsa("[data-identity]", idask).forEach(function (button) {
      button.addEventListener("click", function () {
        setIdentity(button.dataset.identity);
      });
    });
    var otherButton = qs("[data-identity-other]", idask);
    var otherInput = qs("#idask-other", idask);
    if (otherButton && otherInput) {
      otherButton.addEventListener("click", function () { setIdentity(otherInput.value); });
      otherInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter") { event.preventDefault(); setIdentity(otherInput.value); }
      });
    }
  }
  qsa("[data-switch-identity]").forEach(function (button) {
    button.addEventListener("click", askIdentity);
  });

  /* -- toast ------------------------------------------------------------ */
  var toastEl = qs("#toast");
  var toastTimer = null;
  function toast(message, bad) {
    if (!toastEl) return;
    toastEl.textContent = message;
    toastEl.classList.toggle("bad", !!bad);
    toastEl.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toastEl.classList.remove("show"); }, 3200);
  }

  /* -- talking to the server -------------------------------------------- */
  function post(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.assign({ reviewer: reviewer() }, body || {}))
    }).then(function (response) {
      return response.json().then(function (data) {
        if (!response.ok) throw new Error(data.error || "request failed");
        return data;
      });
    });
  }

  function feedbackUrl() {
    return "/api/events/" + encodeURIComponent(WFF.eventId) + "/feedback?" + (WFF.query || "");
  }

  /* -- how far along ----------------------------------------------------
   * The whole score panel is now two facts: how much is checked, and how many
   * faces turned out to be in the wrong pile. Both come off the server with
   * every answer, so the strip can never drift from the journal.
   */
  function applyProgress(progress) {
    if (!progress) return;
    var pct = qs("#rv-pct");
    var fill = qs("#rv-fill");
    var line = qs("#rv-line");
    var bar = qs("#revbar");
    var start = qs("#rv-start");
    if (pct) pct.textContent = progress.percent;
    if (fill) fill.style.width = progress.percent + "%";
    if (line) {
      line.textContent = progress.piles_answered
        ? progress.piles_answered + " of " + progress.piles_total + " piles answered · " +
          progress.faces_fixed + " face" + (progress.faces_fixed === 1 ? "" : "s") + " fixed"
        : "Nothing checked yet. One question per pile, about 10 seconds each.";
    }
    if (bar) bar.classList.toggle("is-done", !!progress.done);
    if (start) {
      start.textContent = progress.done
        ? "Look through them again"
        : progress.piles_answered
          ? "Carry on (" + progress.remaining + " left)"
          : "Start checking";
    }
    var mini = qs("#rv-mini-fill");
    var miniText = qs("#rv-mini-text");
    if (mini) mini.style.width = progress.percent + "%";
    if (miniText) miniText.textContent = progress.piles_answered + " done";
  }

  function applyScore(score) {
    if (!score) return;
    applyProgress(score.progress);
    // Only the expert view still shows this. It is the one number the manual
    // controls exist to move, so it stays live while they are open.
    var advice = qs("#threshold-advice");
    if (advice && score.threshold_summary) {
      advice.innerHTML = "<strong>Where the line should sit:</strong> " + score.threshold_summary;
    }
  }

  /* -- the photo behind a face ------------------------------------------ */
  var modal = qs("#photo-modal");
  function openPhoto(photoId, focusFaceId) {
    if (!modal) return;
    var frame = qs("#photo-frame");
    frame.innerHTML = '<p class="muted small" style="padding:40px">Loading…</p>';
    modal.classList.add("open");
    fetch("/api/events/" + encodeURIComponent(WFF.eventId) + "/photo/" + photoId)
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error("no preview")); })
      .then(function (data) {
        qs("#photo-title").textContent = "Photo " + data.photo_id;
        qs("#photo-path").textContent = data.path || "";
        qs("#photo-note").textContent =
          data.width && data.height
            ? data.width + " x " + data.height + " pixels" +
              (data.taken_at ? " · taken " + data.taken_at.slice(0, 16).replace("T", " ") : "") +
              " · " + data.faces.length + " face(s) found here"
            : "";
        frame.innerHTML = "";
        var img = document.createElement("img");
        img.src = data.url;
        img.alt = "Original photo " + data.photo_id;
        frame.appendChild(img);
        data.faces.forEach(function (face) {
          var box = document.createElement("div");
          box.className = "box" + (face.face_id === focusFaceId ? " focus" : "");
          box.style.left = face.box[0] + "%";
          box.style.top = face.box[1] + "%";
          box.style.width = face.box[2] + "%";
          box.style.height = face.box[3] + "%";
          box.title = face.face_id;
          frame.appendChild(box);
        });
      })
      .catch(function () {
        frame.innerHTML =
          '<p class="muted small" style="padding:40px">No preview saved for this photo.</p>';
      });
  }
  function closePhoto() { if (modal) modal.classList.remove("open"); }
  if (modal) {
    modal.addEventListener("click", function (event) {
      if (event.target === modal || event.target.hasAttribute("data-close-modal")) closePhoto();
    });
  }
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") { closePhoto(); cancelModes(); }
  });

  /* -- two-step modes: "same as another pile" and "belongs to" ---------- */
  var mergeFrom = null;
  var assignFace = null;

  /* Filled in by the review-window block further down. Declared here because
   * the page's click handler is registered before it and has to be able to
   * call into it -- function declarations inside that block are not in scope
   * out here under "use strict". */
  var openReview = function () {};
  var paintPicked = function () {};

  function cancelModes() {
    mergeFrom = null;
    assignFace = null;
    document.body.classList.remove("merging", "assigning");
    qsa(".merge-target").forEach(function (b) { b.hidden = true; });
    qsa(".assign-target").forEach(function (b) { b.hidden = true; });
  }

  function startMerge(personId) {
    cancelModes();
    mergeFrom = personId;
    document.body.classList.add("merging");
    qsa("article.pile").forEach(function (card) {
      if (card.dataset.person !== String(personId)) {
        var button = qs(".merge-target", card);
        if (button) button.hidden = false;
      }
    });
    var source = qs('article.pile[data-person="' + personId + '"]');
    toast("Now press “It’s this person” on the other pile. Escape to cancel.");
    if (source) source.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function startAssign(faceId) {
    cancelModes();
    assignFace = faceId;
    document.body.classList.add("assigning");
    qsa(".assign-target").forEach(function (b) { b.hidden = false; });
    toast("Now press “It’s this person” on the pile it belongs to. Escape to cancel.");
    var piles = qs("#piles");
    if (piles) piles.scrollIntoView({ behavior: "smooth" });
  }

  /* -- the review page --------------------------------------------------
   * Clicking a pile on the page OPENS it in the review window. Clicking a
   * crop inside that window opens the photo it was cut from. The two clicks
   * mean different things on purpose: the page is for choosing what to look
   * at, the window is for looking.
   */
  var peopleRoot = qs("#people");
  if (peopleRoot || qs("#leftover-grid")) {
    document.addEventListener("click", function (event) {
      var button = event.target.closest("[data-action]");
      var faceEl = event.target.closest(".face");
      var card = event.target.closest("article.pile");
      var personId = card ? parseInt(card.dataset.person, 10) : null;

      var slot = qs("#rv-slot");
      var inWindow = function (node) { return !!(slot && node && slot.contains(node)); };

      if (!button) {
        if (faceEl && document.body.classList.contains("rv-fixing") && inWindow(faceEl)) {
          faceEl.classList.toggle("picked");
          paintPicked();
          return;
        }
        // On the page, clicking a pile opens it for checking. Inside the
        // window -- and for the leftovers, which are in no pile -- a crop
        // opens the photo it was cut from.
        if (card && !inWindow(card) &&
            !document.body.classList.contains("merging") &&
            !document.body.classList.contains("assigning")) {
          openReview(card);
          return;
        }
        if (faceEl) openPhoto(faceEl.dataset.photo, faceEl.dataset.face);
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      var action = button.dataset.action;

      if (action === "not-a-face") {
        post(feedbackUrl(), {
          kind: "not_a_face",
          face_id: faceEl.dataset.face,
          person_id: personId
        })
          .then(function (data) {
            faceEl.classList.add("notface");
            var tools = qs(".tools", faceEl);
            if (tools) {
              tools.innerHTML =
                '<button class="undo" data-action="undo" data-record="' + data.record_id + '">undo</button>';
            }
            applyScore(data.score);
            toast("Marked as not a face.");
          })
          .catch(function (err) { toast(err.message, true); });
        return;
      }

      if (action === "undo") {
        post("/api/events/" + encodeURIComponent(WFF.eventId) + "/feedback/" + button.dataset.record + "/retract?" + (WFF.query || ""))
          .then(function (data) {
            faceEl.classList.remove("flagged", "notface");
            faceEl.dataset.record = "";
            var tools = qs(".tools", faceEl);
            if (tools) {
              tools.innerHTML =
                '<button class="assign" data-action="assign-start">who is this?</button>' +
                '<button data-action="not-a-face">not a face</button>';
            }
            applyScore(data.score);
            toast("Took that back.");
          })
          .catch(function (err) { toast(err.message, true); });
        return;
      }

      if (action === "merge-start") { startMerge(personId); return; }

      if (action === "merge-finish") {
        if (mergeFrom === null) return;
        post(feedbackUrl(), { kind: "same_person", person_a: mergeFrom, person_b: personId })
          .then(function (data) {
            applyScore(data.score);
            cancelModes();
            toast("Saved: those two piles are the same person.");
          })
          .catch(function (err) { toast(err.message, true); cancelModes(); });
        return;
      }

      if (action === "assign-start") { startAssign(faceEl.dataset.face); return; }

      if (action === "assign-finish") {
        if (!assignFace) return;
        post(feedbackUrl(), { kind: "face_belongs", face_id: assignFace, person_id: personId })
          .then(function (data) {
            var moved = qs('.face[data-face="' + assignFace + '"]');
            if (moved) moved.classList.add("judged");
            applyScore(data.score);
            cancelModes();
            toast("Saved: that face belongs to this person.");
          })
          .catch(function (err) { toast(err.message, true); cancelModes(); });
        return;
      }

      if (action === "note") {
        var about = card ? qs("h4", card).textContent : "this run";
        var text = window.prompt("Note about " + about + ":");
        if (!text) return;
        post(feedbackUrl(), { kind: "note", text: text, person_id: personId })
          .then(function () { toast("Note saved. It shows up at the bottom of the page."); })
          .catch(function (err) { toast(err.message, true); });
      }
    });
  }

  /* ====================================================================
   * THE REVIEW WINDOW
   * One pile, one question, two keys. The pile is MOVED into the window
   * rather than copied: the page already holds every crop of every pile, and
   * cloning 241 <img> tags to show the same faces again would double the DOM
   * of the biggest folder we have for no gain.
   * ==================================================================== */
  var rvWindow = qs("#rv-window");
  var rvSlot = qs("#rv-slot");
  var pilesRoot = qs("#piles");

  if (rvWindow && rvSlot && pilesRoot) {
    var rvPlaceholder = null;
    var rvCurrent = null;
    var footAsk = qs("#rv-foot-ask");
    var footFix = qs("#rv-foot-fix");
    var endPanel = qs("#rv-end");

    /* Captured once, in page order. NOT re-queried from #piles: the pile being
     * reviewed has been moved out of it, so a live query would lose exactly
     * the one the window is showing -- and with it the "pile 9 of 20". */
    var pileList = qsa("article.pile", pilesRoot);
    var allPiles = function () { return pileList; };
    var answeredCount = function () {
      return pileList.filter(function (p) { return p.dataset.answered === "1"; }).length;
    };

    function paintHead(pile) {
      var piles = allPiles();
      var index = piles.indexOf(pile);
      var sub = qs("#rv-sub");
      if (sub) {
        sub.textContent =
          "Pile " + (index + 1) + " of " + piles.length + " · " +
          pile.dataset.count + " faces from " + pile.dataset.photos + " photos";
      }
      var question = qs("#rv-question");
      if (question) question.textContent = "Is everyone here the same person?";
      var mini = qs("#rv-mini-fill");
      var miniText = qs("#rv-mini-text");
      var done = answeredCount();
      if (mini) mini.style.width = (piles.length ? (done / piles.length) * 100 : 0) + "%";
      if (miniText) miniText.textContent = done + " done";
    }

    function paintFixCount() {
      var picked = qsa(".face.picked", rvSlot).length;
      var node = qs("#rv-fix-count");
      if (!node) return;
      node.textContent = picked
        ? picked + " face" + (picked === 1 ? "" : "s") + " marked as somebody else"
        : "Tap every face that isn’t this person.";
    }
    paintPicked = paintFixCount;

    function stopFixing() {
      document.body.classList.remove("rv-fixing");
      qsa(".face.picked", rvSlot).forEach(function (f) { f.classList.remove("picked"); });
      if (footAsk) footAsk.hidden = false;
      if (footFix) footFix.hidden = true;
    }

    function putBack() {
      if (rvCurrent && rvPlaceholder && rvPlaceholder.parentNode) {
        rvPlaceholder.parentNode.replaceChild(rvCurrent, rvPlaceholder);
      }
      rvCurrent = null;
      rvPlaceholder = null;
    }

    function open(pile) {
      if (!pile) return;
      stopFixing();
      putBack();
      if (endPanel) endPanel.hidden = true;
      if (footAsk) footAsk.hidden = false;
      rvPlaceholder = document.createElement("div");
      rvPlaceholder.className = "pile-holder";
      pile.parentNode.replaceChild(rvPlaceholder, pile);
      rvSlot.appendChild(pile);
      rvSlot.hidden = false;
      rvSlot.scrollTop = 0;
      rvCurrent = pile;
      paintHead(pile);
      rvWindow.hidden = false;
      document.body.classList.add("rv-open");
    }
    openReview = open;

    function close() {
      stopFixing();
      putBack();
      rvWindow.hidden = true;
      document.body.classList.remove("rv-open");
    }

    function nextUnanswered(after) {
      var piles = allPiles();
      var start = after ? piles.indexOf(after) + 1 : 0;
      for (var i = start; i < piles.length; i += 1) {
        if (piles[i].dataset.answered !== "1") return piles[i];
      }
      // Wrap around: something skipped earlier is still unanswered.
      for (var j = 0; j < start && j < piles.length; j += 1) {
        if (piles[j].dataset.answered !== "1") return piles[j];
      }
      return null;
    }

    function showEnd() {
      var piles = allPiles();
      var fixed = piles.filter(function (p) { return p.classList.contains("fixed"); }).length;
      putBack();
      rvSlot.hidden = true;
      if (footAsk) footAsk.hidden = true;
      if (footFix) footFix.hidden = true;
      if (endPanel) endPanel.hidden = false;
      var title = qs("#rv-end-title");
      var line = qs("#rv-end-line");
      if (title) {
        title.textContent = piles.length === 1
          ? "That was the only pile."
          : "All " + piles.length + " piles checked.";
      }
      if (line) {
        line.textContent = fixed
          ? fixed + " pile" + (fixed === 1 ? "" : "s") + " had somebody else mixed in. Every answer is now part of the answer key this run is scored against."
          : "Nothing was in the wrong pile. Every answer is now part of the answer key this run is scored against.";
      }
      var sub = qs("#rv-sub");
      if (sub) sub.textContent = "Nothing left to check";
      var question = qs("#rv-question");
      if (question) question.textContent = "Done";
    }

    function advance(from) {
      var next = nextUnanswered(from);
      if (next) { open(next); } else { showEnd(); }
    }

    function markAnswered(pile, fixedFaces) {
      pile.dataset.answered = "1";
      pile.classList.add("answered");
      pile.classList.toggle("fixed", fixedFaces > 0);
      var state = qs(".pile-state", pile);
      if (state) {
        state.textContent = fixedFaces
          ? fixedFaces + " fixed"
          : "checked";
      }
    }

    function answerYes() {
      var pile = rvCurrent;
      if (!pile) return;
      var personId = parseInt(pile.dataset.person, 10);
      post(feedbackUrl(), { kind: "person_ok", person_id: personId })
        .then(function (data) {
          pile.dataset.record = data.record_id;
          markAnswered(pile, qsa(".face.flagged", pile).length);
          applyScore(data.score);
          advance(pile);
        })
        .catch(function (err) { toast(err.message, true); });
    }

    function startFixing() {
      if (!rvCurrent) return;
      document.body.classList.add("rv-fixing");
      // Faces already marked wrong start selected, so the same screen both
      // adds and takes back -- otherwise unmarking would need a second UI.
      qsa(".face.flagged", rvSlot).forEach(function (f) { f.classList.add("picked"); });
      if (footAsk) footAsk.hidden = true;
      if (footFix) footFix.hidden = false;
      paintFixCount();
    }

    function finishFixing() {
      var pile = rvCurrent;
      if (!pile) return;
      var personId = parseInt(pile.dataset.person, 10);
      var picked = qsa(".face.picked", rvSlot);
      var faces = qsa(".face", rvSlot);
      var added = picked.filter(function (f) { return !f.dataset.record; });
      var removed = faces.filter(function (f) {
        return f.dataset.record && !f.classList.contains("picked");
      });
      if (!added.length && !removed.length) {
        stopFixing();
        toast("Nothing marked — nothing saved.");
        return;
      }

      var chain = Promise.resolve();
      added.forEach(function (face) {
        chain = chain.then(function () {
          return post(feedbackUrl(), {
            kind: "face_wrong",
            face_id: face.dataset.face,
            person_id: personId
          }).then(function (data) {
            face.dataset.record = data.record_id;
            face.classList.add("flagged");
          });
        });
      });
      removed.forEach(function (face) {
        chain = chain.then(function () {
          return post(
            "/api/events/" + encodeURIComponent(WFF.eventId) +
            "/feedback/" + face.dataset.record + "/retract?" + (WFF.query || "")
          ).then(function () {
            face.dataset.record = "";
            face.classList.remove("flagged");
          });
        });
      });

      // Then say the rest of the pile IS one person. Order matters: the server
      // leaves flagged faces out of an approval, so flagging first and
      // approving second records "all one person except these" in one go.
      // Approving first and flagging second produced a contradiction.
      chain
        .then(function () {
          return post(feedbackUrl(), { kind: "person_ok", person_id: personId })
            .catch(function () { return null; });  // <2 faces left is not a failure
        })
        .then(function (data) {
          stopFixing();
          markAnswered(pile, qsa(".face.flagged", pile).length);
          if (data && data.score) {
            applyScore(data.score);
            advance(pile);
            return;
          }
          return fetch(
            "/api/events/" + encodeURIComponent(WFF.eventId) + "/score?" + (WFF.query || "")
          )
            .then(function (r) { return r.json(); })
            .then(function (score) { applyScore(score); advance(pile); });
        })
        .catch(function (err) { toast(err.message, true); stopFixing(); });
    }

    qsa("[data-rv]").forEach(function (button) {
      button.addEventListener("click", function (event) {
        event.preventDefault();
        var what = button.dataset.rv;
        if (what === "close") close();
        else if (what === "yes") answerYes();
        else if (what === "no") startFixing();
        else if (what === "skip") advance(rvCurrent);
        else if (what === "fix-done") finishFixing();
        else if (what === "fix-cancel") stopFixing();
      });
    });

    var startButton = qs("#rv-start");
    if (startButton) {
      startButton.addEventListener("click", function () {
        var next = nextUnanswered(null);
        if (next) { open(next); } else { open(allPiles()[0]); }
      });
    }

    rvWindow.addEventListener("click", function (event) {
      if (event.target === rvWindow) close();
    });

    document.addEventListener("keydown", function (event) {
      if (rvWindow.hidden) return;
      if (modal && modal.classList.contains("open")) return;
      if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") return;
      var key = event.key.toLowerCase();
      if (event.key === "Escape") { close(); return; }
      if (document.body.classList.contains("rv-fixing")) {
        if (event.key === "Enter") { event.preventDefault(); finishFixing(); }
        return;
      }
      if (key === "y") { event.preventDefault(); answerYes(); }
      else if (key === "n") { event.preventDefault(); startFixing(); }
      else if (event.key === "ArrowRight") { event.preventDefault(); advance(rvCurrent); }
    });

    // Every pile starts with its tick and its "checked" caption already right,
    // rendered from the journal rather than assumed.
    allPiles().forEach(function (pile) {
      if (pile.dataset.answered === "1") {
        var fixed = qsa(".face.flagged", pile).length;
        pile.classList.add("answered");
        pile.classList.toggle("fixed", fixed > 0);
        var state = qs(".pile-state", pile);
        if (state) state.textContent = fixed ? fixed + " fixed" : "checked";
      }
      var count = parseInt(pile.dataset.count, 10) || 0;
      var more = qs(".more", pile);
      if (more) {
        more.textContent = count > 14
          ? "click to see all " + count + " and check this pile"
          : "click to check this pile";
      }
    });
  }

  /* -- show only the faces mistakes hide in ------------------------------ */
  var riskyToggle = qs("#risky-toggle");
  if (riskyToggle) {
    var smallFaces = qsa(".pile .face.risky").length;
    var totalFaces = qsa(".pile .face").length;
    var pctSmall = totalFaces ? Math.round((smallFaces / totalFaces) * 100) : 0;
    riskyToggle.textContent = "Show the risky ones (" + pctSmall + "%)";
    riskyToggle.addEventListener("click", function () {
      var on = document.body.classList.toggle("risky");
      riskyToggle.classList.toggle("primary", on);
      riskyToggle.textContent = on
        ? "Show all faces again"
        : "Show the risky ones (" + pctSmall + "%)";
      if (on) {
        toast(smallFaces + " faces are under 112px tall in the photo — that is where nearly every mistake is.");
      }
    });
  }

  /* -- renaming a run, and saying who ran it -----------------------------
   * Both edit in place: click, type, Enter. Nothing on disk moves -- the id
   * stays, and it is still printed beside a renamed folder.
   */
  document.addEventListener("click", function (event) {
    var button = event.target.closest('[data-action="rename"], [data-action="set-who"]');
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    var isName = button.dataset.action === "rename";
    var host = button.closest("[data-event]") || button.closest(".review");
    var eventId = (host && host.dataset.event) || WFF.eventId;
    if (!eventId) return;

    var target = isName
      ? (button.closest(".head, .review-top") || host).querySelector('[data-role="name"]')
      : button.closest('[data-role="who"]');
    if (!target || target.querySelector(".nameedit")) return;

    var original = target.innerHTML;
    var current = isName ? target.textContent.trim() : "";
    var box = document.createElement("input");
    box.type = "text";
    box.className = "nameedit";
    box.maxLength = isName ? 60 : 40;
    box.value = current;
    box.placeholder = isName ? "a name you'll recognise" : "Adarsh, Devesh, …";
    target.textContent = "";
    target.appendChild(box);
    box.focus();
    box.select();

    // Enter saves and then blurs, which would save a second time.
    var settled = false;
    var finish = function (save) {
      if (settled) return;
      settled = true;
      var value = box.value.trim();
      target.innerHTML = original;
      if (!save) return;
      // `tested_by`, not `reviewer`: post() attaches `reviewer` to every
      // request as "who is doing this", and the two must not be the same key.
      var body = isName ? { name: value } : { tested_by: value };
      post("/api/events/" + encodeURIComponent(eventId) + "/label", body)
        .then(function () {
          toast(isName
            ? (value ? "Renamed to “" + value + "”." : "Name cleared — back to the folder id.")
            : (value ? "Recorded: " + value + " ran this." : "Cleared."));
          window.location.reload();
        })
        .catch(function (err) { toast(err.message, true); });
    };

    box.addEventListener("keydown", function (keyEvent) {
      if (keyEvent.key === "Enter") { keyEvent.preventDefault(); finish(true); }
      if (keyEvent.key === "Escape") { keyEvent.preventDefault(); finish(false); }
    });
    box.addEventListener("blur", function () { finish(true); });
  });

  /* -- the one-time Google key ------------------------------------------
   * Pasting a Drive link is the whole interaction we promise, so the key that
   * makes Drive work has to be settable here. It is checked against the very
   * folder that was just pasted before it is saved -- and when it works, the
   * run that was interrupted starts on its own, because being sent back to
   * re-paste the same link is exactly the friction this removes.
   */
  var keybox = qs("#keybox");
  if (keybox) {
    var keyPanel = qs("#keypanel");
    var keyInput = qs("#key-input");
    var keyResult = qs("#key-result");
    var keySaveBtn = qs('[data-action="key-save"]', keybox);

    var openKeyPanel = function () {
      keyPanel.hidden = false;
      keybox.classList.add("open");
      if (keyInput) keyInput.focus();
    };

    qsa('[data-action="key-open"]', keybox).forEach(function (button) {
      button.addEventListener("click", openKeyPanel);
    });

    var sayKeyResult = function (text, bad) {
      keyResult.textContent = text;
      keyResult.className = "small " + (bad ? "keybad" : "keygood");
    };

    var startPendingRun = function () {
      var link = keybox.dataset.link;
      if (!link) return false;
      // Submit the real form rather than fetch(): /runs redirects to the run's
      // own page, which is where someone who just pressed Start expects to be.
      var form = document.createElement("form");
      form.method = "post";
      form.action = "/runs";
      [["link", link], ["event_id", keybox.dataset.eventId || ""],
       ["reviewer", reviewer()]].forEach(function (pair) {
        var field = document.createElement("input");
        field.type = "hidden";
        field.name = pair[0];
        field.value = pair[1];
        form.appendChild(field);
      });
      document.body.appendChild(form);
      form.submit();
      return true;
    };

    var saveKey = function () {
      var key = (keyInput.value || "").trim();
      if (!key) { sayKeyResult("Paste the key first.", true); return; }
      keySaveBtn.disabled = true;
      sayKeyResult("Checking it with Google...", false);
      fetch("/api/settings/google-key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: key, link: keybox.dataset.link || "" })
      })
        .then(function (response) { return response.json(); })
        .then(function (data) {
          keySaveBtn.disabled = false;
          if (data.ok) {
            keyInput.value = "";
            sayKeyResult(data.message, false);
            if (startPendingRun()) return;
            toast("Google Drive is set up.");
            setTimeout(function () { window.location.reload(); }, 900);
            return;
          }
          // A key that Google accepted is kept even when the FOLDER is the
          // problem -- otherwise fixing the sharing would mean re-pasting a
          // key that was never wrong.
          sayKeyResult(data.message + (data.key_kept ? " (Your key is saved and working.)" : ""),
                       true);
        })
        .catch(function () {
          keySaveBtn.disabled = false;
          sayKeyResult("Could not reach the console. Is it still running?", true);
        });
    };

    if (keySaveBtn) keySaveBtn.addEventListener("click", saveKey);
    if (keyInput) {
      keyInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter") { event.preventDefault(); saveKey(); }
      });
    }
  }

  /* -- the projection calculator ---------------------------------------
   * The server owns the arithmetic. It knows whether the speed is a measured
   * average or the recorded fallback, and the page must never quote a number
   * the run journal does not support.
   */
  var calcInput = qs("#calc-photos");
  if (calcInput) {
    var calcOut = qs("#calc-out");
    var calcTimer = null;

    var runForecast = function () {
      var count = parseInt(calcInput.value.replace(/[^0-9]/g, ""), 10);
      if (!count) {
        calcOut.textContent = "—";
        return;
      }
      fetch("/api/forecast?photos=" + count)
        .then(function (r) { return r.json(); })
        .then(function (data) { calcOut.textContent = data.text; })
        .catch(function () { calcOut.textContent = "—"; });
    };

    calcInput.addEventListener("input", function () {
      clearTimeout(calcTimer);
      calcTimer = setTimeout(runForecast, 220);
    });
    calcInput.addEventListener("blur", function () {
      var count = parseInt(calcInput.value.replace(/[^0-9]/g, ""), 10);
      if (count) calcInput.value = count.toLocaleString("en-US");
    });
  }

  /* -- finding a folder in the run list ---------------------------------
   * The list only grows, so it lives in a fixed-height scroll box. Once it is
   * longer than the box, scrolling to find one folder by name is the thing you
   * do most, so filter in place -- purely client-side, nothing to reload.
   */
  var runFilter = qs("#run-filter");
  if (runFilter) {
    var runEmpty = qs("#run-filter-empty");
    var runScroller = qs("#run-scroller");
    runFilter.addEventListener("input", function () {
      var needle = runFilter.value.trim().toLowerCase();
      var shown = 0;
      qsa("#run-scroller .runcard").forEach(function (card) {
        var hit = !needle || (card.dataset.name || "").indexOf(needle) !== -1;
        card.hidden = !hit;
        if (hit) shown += 1;
      });
      // A size heading with nothing under it reads as "no large runs exist",
      // which is a different statement from "none of them match what you typed".
      qsa("#run-scroller .sizegroup").forEach(function (group) {
        group.hidden = qsa(".runcard", group).every(function (card) { return card.hidden; });
      });
      if (runEmpty) runEmpty.hidden = shown !== 0;
      if (runScroller) runScroller.scrollTop = 0;
    });
  }

  /* -- the home page refreshes itself while something is running -------- */
  if (WFF.home && qs(".runcard.live")) {
    setTimeout(function () { window.location.reload(); }, 5000);
  }

  /* -- threshold sliders ------------------------------------------------ */
  ["p1", "p2", "minf"].forEach(function (name) {
    var input = qs("#" + name);
    var label = qs("#" + name + "-value");
    if (input && label) {
      input.addEventListener("input", function () { label.textContent = input.value; });
    }
  });

  /* -- watching a run happen --------------------------------------------
   * One poll drives all of it: the step list, the counters and the wall of
   * crops. They are read from the same journal on the server, so what you see
   * can never be a face count from one moment beside a photo count from
   * another.
   */
  var liveRoot = qs("#live");
  if (liveRoot) {
    var liveEvent = liveRoot.dataset.event;
    var liveJob = liveRoot.dataset.job;
    var wall = qs("#live-faces");
    var seen = {};
    qsa(".fc", wall).forEach(function (node) { seen[node.dataset.face] = true; });

    var stopButton = qs("#job-stop");
    if (stopButton) {
      stopButton.addEventListener("click", function () {
        stopButton.disabled = true;
        post("/api/jobs/" + liveJob + "/stop").then(function () {
          toast("Stopping after the current photo. Nothing done so far is lost.");
        });
      });
    }

    var setText = function (sel, value) {
      var node = qs(sel);
      if (node) node.textContent = value;
    };

    var renderSteps = function (steps) {
      var host = qs("#live-steps");
      if (!host || !steps) return;
      host.textContent = "";
      steps.forEach(function (step, index) {
        var row = document.createElement("div");
        row.className = "step " + step.state;

        var mark = document.createElement("span");
        mark.className = "mark";
        mark.textContent =
          step.state === "done" ? "✓" :
          step.state === "now" ? "›" :
          step.state === "failed" ? "!" : String(index + 1);

        var body = document.createElement("div");
        var title = document.createElement("div");
        title.className = "t";
        title.textContent = step.title;
        var detail = document.createElement("div");
        detail.className = "d";
        detail.textContent = step.detail;
        body.appendChild(title);
        body.appendChild(detail);

        if (typeof step.percent === "number") {
          var bar = document.createElement("div");
          bar.className = "progress thin";
          bar.style.marginTop = "8px";
          var fill = document.createElement("span");
          fill.style.width = step.percent + "%";
          bar.appendChild(fill);
          body.appendChild(bar);
        }
        row.appendChild(mark);
        row.appendChild(body);
        host.appendChild(row);
      });
    };

    var addFaces = function (faces) {
      if (!wall || !faces || !faces.length) return;
      // Newest first, so they are prepended in reverse to keep the order.
      faces.slice().reverse().forEach(function (face) {
        if (seen[face.face_id]) return;
        seen[face.face_id] = true;
        var tile = document.createElement("div");
        tile.className = "fc" + (face.small ? " small" : "");
        tile.dataset.face = face.face_id;
        var img = document.createElement("img");
        img.loading = "lazy";
        img.width = 72;
        img.height = 72;
        img.alt = "";
        img.src = face.url;
        var px = document.createElement("span");
        px.className = "px";
        px.textContent = face.height_px;
        tile.appendChild(img);
        tile.appendChild(px);
        wall.insertBefore(tile, wall.firstChild);
      });
      var extra = wall.children.length - 60;
      for (var i = 0; i < extra; i += 1) wall.removeChild(wall.lastChild);
      var empty = qs("#live-empty");
      if (empty) empty.hidden = wall.children.length > 0;
    };

    var poll = function () {
      fetch("/api/events/" + encodeURIComponent(liveEvent) + "/live?job=" + encodeURIComponent(liveJob))
        .then(function (r) { return r.json(); })
        .then(function (data) {
          var job = data.job;
          renderSteps(data.steps);
          addFaces(data.faces);

          var totals = data.totals || {};
          setText("#live-detected", (totals.detected || 0).toLocaleString());
          setText("#live-accepted", (totals.accepted || 0).toLocaleString());
          setText("#live-small", (totals.too_small || 0).toLocaleString());
          setText("#live-usable", totals.usable_pct || 0);
          setText("#live-nofaces", (totals.photos_without_faces || 0).toLocaleString());
          setText("#live-biggest", totals.biggest_photo || 0);

          if (job) {
            setText("#live-done", job.photos_done.toLocaleString());
            setText("#live-message", job.message);
            setText("#live-rate", job.seconds_per_photo || "—");
            setText("#job-phase-text", job.phase);
            setText(
              "#live-eta",
              job.percent + "% done" +
                (job.eta_seconds ? " · about " + humanLeft(job.eta_seconds) + " left" : "")
            );
            var bar = qs("#job-bar");
            if (bar) bar.style.width = job.percent + "%";
            var log = qs("#job-log");
            if (log) {
              var atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
              log.textContent = job.log.join("\n");
              if (atBottom) log.scrollTop = log.scrollHeight;
            }
            var phase = qs("#job-phase");
            if (phase) {
              phase.className =
                "phase " + (["done", "failed", "stopped"].indexOf(job.phase) >= 0 ? job.phase : "");
            }
            if (job.finished) {
              // Piles only exist once grouping has run, so reload to show them.
              toast("Finished — loading the people it found.");
              setTimeout(function () { window.location.reload(); }, 900);
              return;
            }
          }
          setTimeout(poll, 2000);
        })
        .catch(function () { setTimeout(poll, 5000); });
    };
    setTimeout(poll, 1200);
  }

  function humanLeft(seconds) {
    seconds = Math.round(seconds || 0);
    if (seconds < 60) return seconds + "s";
    if (seconds < 3600) return Math.round(seconds / 60) + " min";
    return Math.floor(seconds / 3600) + "h " + Math.round((seconds % 3600) / 60) + "m";
  }

  /* -- judge mode ------------------------------------------------------- */
  var judgeRoot = qs("#judge");
  if (judgeRoot) {
    var dataNode = qs("#judge-data");
    var pairs = dataNode ? JSON.parse(dataNode.textContent) : [];
    var cursor = 0;
    var answered = { same: 0, different: 0, skipped: 0 };

    var cropUrl = function (faceId) {
      return "/i/" + encodeURIComponent(WFF.eventId) + "/crop/" + faceId;
    };

    function render() {
      if (cursor >= pairs.length) {
        qs(".judge-pair").hidden = true;
        qs(".judge-actions").hidden = true;
        qs("#judge-progress").textContent = "Done";
        var done = qs("#judge-done");
        done.hidden = false;
        qs("#judge-summary").textContent =
          answered.same + " same, " + answered.different + " different, " +
          answered.skipped + " skipped. Every one of those is now part of the answer key.";
        return;
      }
      var pair = pairs[cursor];
      qs("#judge-a").src = cropUrl(pair.a.face_id);
      qs("#judge-b").src = cropUrl(pair.b.face_id);
      qs("#judge-a").dataset.photo = pair.a.photo_id;
      qs("#judge-a").dataset.face = pair.a.face_id;
      qs("#judge-b").dataset.photo = pair.b.photo_id;
      qs("#judge-b").dataset.face = pair.b.face_id;
      qs("#judge-a-cap").textContent =
        pair.a.person + " · " + pair.a.faces_in_pile + " faces · " + pair.a.height_px + "px";
      qs("#judge-b-cap").textContent =
        pair.b.person + " · " + pair.b.faces_in_pile + " faces · " + pair.b.height_px + "px";
      qs("#judge-index").textContent = cursor + 1;
      qs("#judge-guess").textContent = pair.verdict === "same" ? "same person" : "different people";
      qs("#judge-distance").textContent = "(" + pair.distance + ")";
      qs("#judge-bar").style.width = (cursor / pairs.length) * 100 + "%";
    }

    function answer(verdict) {
      var pair = pairs[cursor];
      if (!pair) return;
      if (verdict === "skip") {
        answered.skipped += 1;
        cursor += 1;
        render();
        return;
      }
      answered[verdict] += 1;
      post(feedbackUrl(), {
        kind: "pair",
        face_a: pair.a.face_id,
        face_b: pair.b.face_id,
        same: verdict === "same"
      })
        .then(function (data) { applyScore(data.score); })
        .catch(function (err) { toast(err.message, true); });
      cursor += 1;
      render();
    }

    qsa("[data-judge]").forEach(function (button) {
      button.addEventListener("click", function () { answer(button.dataset.judge); });
    });
    ["#judge-a", "#judge-b"].forEach(function (sel) {
      var img = qs(sel);
      img.style.cursor = "zoom-in";
      img.addEventListener("click", function () {
        openPhoto(img.dataset.photo, img.dataset.face);
      });
    });
    document.addEventListener("keydown", function (event) {
      if (modal && modal.classList.contains("open")) return;
      if (event.target.tagName === "INPUT") return;
      var key = event.key.toLowerCase();
      if (key === "s") { answer("same"); }
      else if (key === "d") { answer("different"); }
      else if (event.key === " ") { event.preventDefault(); answer("skip"); }
    });
    render();
  }
})();
