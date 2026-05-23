"""
jetson_a.py  —  Jetson Nano A: board vision + move detection + cloud POST
---------------------------------------------------------------------------
DEPENDENCIES
    pip install opencv-python python-chess requests numpy

HARDWARE
    Camera mounted directly above the board, centred.
    Any USB webcam or CSI camera supported by OpenCV.

USAGE
    python3 jetson_a.py
    Press Enter after each human move.
    Press Ctrl+C to quit (game state is saved to game_state.txt).
"""

import cv2
import chess
import numpy as np
import requests
import os
import sys

# ---------------------------------------------------------------------------
# CONFIGURATION  —  edit these before running
# ---------------------------------------------------------------------------

CLOUD_URL      = "https://chess-engine-694749521649.us-central1.run.app/get_move"

CAMERA_INDEX   = 0                                    # USB camera (0 = first)
BOARD_PX       = 800                                  # warped board size (px)
DIFF_THRESHOLD = 25                                   # pixel diff sensitivity
                                                      # raise if false triggers,
                                                      # lower if misses changes
STATE_FILE     = "game_state.txt"                     # crash-recovery file
SNAPSHOT_FILE  = "last_snapshot.png"                  # persisted between runs

# ---------------------------------------------------------------------------
# BOARD STATE
# ---------------------------------------------------------------------------

def load_board() -> chess.Board:
    """Load board from saved FEN, or start a fresh game."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            fen = f.read().strip()
        if fen:
            print(f"[INFO] Resuming game from {STATE_FILE}")
            return chess.Board(fen)
    print("[INFO] Starting new game")
    return chess.Board()


def save_board(board: chess.Board):
    """Persist current FEN to disk so we can recover from a crash."""
    with open(STATE_FILE, "w") as f:
        f.write(board.fen())


# ---------------------------------------------------------------------------
# CAMERA + PERSPECTIVE WARP
# ---------------------------------------------------------------------------

def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open USB camera at index {index}. "
                 "Try index 1 or 2 if you have multiple cameras.")
    return cap


def capture_frame(cap: cv2.VideoCapture) -> np.ndarray:
    ret, frame = cap.read()
    if not ret:
        sys.exit("[ERROR] Failed to read frame from camera")
    return frame


def show_preview(cap: cv2.VideoCapture, board: "chess.Board"):
    """
    Stream live camera frames in a window until the user presses Enter.
    Overlays whose turn it is onto the frame so you can confirm the board
    looks correct before capturing. Press Enter in the terminal or in the
    OpenCV window to take the snapshot.
    """
    import select
    turn_text = "White's turn" if board.turn == chess.WHITE else "Black's turn"
    print(f"   [{turn_text}] Adjust board if needed, then press Enter to capture...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Semi-transparent bar at top
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"{turn_text}  |  Press Enter to capture",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1,
                    cv2.LINE_AA)

        # Draw green outline around detected board corners
        corners = find_board_corners(frame)
        if corners is not None:
            ordered = order_corners(corners)
            pts = ordered.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]),
                         (0, 255, 100), 2)

        cv2.imshow("Chess Vision — Jetson A", frame)

        # Enter pressed inside the CV window
        key = cv2.waitKey(30) & 0xFF
        if key == 13:
            break

        # Enter pressed in the terminal (non-blocking stdin check)
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            break


def find_board_corners(frame: np.ndarray):
    """
    Detect the four corners of the chess board in the raw camera frame.
    Returns a (4,2) float32 array ordered [top-left, top-right,
    bottom-right, bottom-left], or None if the board can't be found.

    HOW IT WORKS
        1. Convert to grayscale and blur to reduce noise.
        2. Canny edge detection to find strong edges.
        3. Find all external contours.
        4. Look for the largest contour that has exactly 4 corners
           (i.e. a quadrilateral) — that's the board border.

    TIPS FOR RELIABLE DETECTION
        - Use a board with a clear contrasting border (dark frame on
          light table, or vice versa).
        - Keep lighting consistent and avoid harsh shadows across corners.
        - If detection fails often, increase blur_ksize or lower
          canny_low to pick up softer edges.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Re-order four corner points to [top-left, top-right,
    bottom-right, bottom-left] regardless of how they came out of
    findContours.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left  (smallest x+y)
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right (smallest x-y)
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def warp_board(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Apply a perspective transform so the board fills a square image
    of BOARD_PX × BOARD_PX pixels, perfectly top-down.
    """
    ordered = order_corners(corners)
    dst = np.array([
        [0,          0],
        [BOARD_PX-1, 0],
        [BOARD_PX-1, BOARD_PX-1],
        [0,          BOARD_PX-1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(frame, M, (BOARD_PX, BOARD_PX))


def get_warped_snapshot(cap: cv2.VideoCapture) -> np.ndarray | None:
    """
    Capture a frame, detect the board, and return the warped image.
    Returns None and prints a warning if the board can't be found.
    """
    frame = capture_frame(cap)
    corners = find_board_corners(frame)
    if corners is None:
        print("[WARN] Could not detect board corners in this frame.")
        print("       Check lighting, camera angle, and board border contrast.")
        return None
    return warp_board(frame, corners)


# ---------------------------------------------------------------------------
# SQUARE DIFFING
# ---------------------------------------------------------------------------

SQ_PX = BOARD_PX // 8   # pixels per square (100 if BOARD_PX=800)


def square_crops(warped: np.ndarray) -> dict:
    """
    Slice the warped board into 64 crops keyed by (file, rank)
    where file 0 = a-file, rank 0 = rank 1 (White's back rank).

    IMPORTANT: rank is counted from the BOTTOM of the image because
    chess rank 1 is at the bottom, but image row 0 is at the top.
    The conversion is:  image_row = (7 - rank)
    """
    crops = {}
    for rank in range(8):
        for file in range(8):
            row = 7 - rank                   # flip: rank 1 → bottom of image
            x = file * SQ_PX
            y = row  * SQ_PX
            crops[(file, rank)] = warped[y : y + SQ_PX, x : x + SQ_PX]
    return crops


def find_changed_squares(
    before: np.ndarray,
    after:  np.ndarray,
) -> list[tuple[int, int]]:
    """
    Compare two warped board images square by square.
    Returns a list of (file, rank) tuples for squares whose mean
    absolute pixel difference exceeds DIFF_THRESHOLD.

    A normal move changes exactly 2 squares.
    Castling changes 4 (king + rook both move).
    """
    crops_before = square_crops(before)
    crops_after  = square_crops(after)
    changed = []
    for sq, crop_after in crops_after.items():
        diff = cv2.absdiff(crop_after, crops_before[sq])
        if diff.mean() > DIFF_THRESHOLD:
            changed.append(sq)
    return changed


# ---------------------------------------------------------------------------
# MOVE INFERENCE
# ---------------------------------------------------------------------------

def coords_to_square(file: int, rank: int) -> chess.Square:
    """Convert (file 0-7, rank 0-7) to a python-chess Square integer."""
    return chess.square(file, rank)


def infer_move(
    board: chess.Board,
    changed: list[tuple[int, int]],
) -> chess.Move | None:
    """
    Given 2 (or 4 for castling) changed squares and the current board,
    find the legal move that matches.

    STRATEGY
        For a 2-square change: try both directions (A→B and B→A).
        Also tries pawn promotion to queen automatically.
        For a 4-square change: look for a castling move among legal moves.
        If no legal move matches, return None.
    """
    squares = [coords_to_square(f, r) for f, r in changed]

    # --- castling: 4 changed squares ---
    if len(squares) == 4:
        for move in board.legal_moves:
            if board.is_castling(move):
                return move
        return None

    # --- normal move: 2 changed squares ---
    if len(squares) != 2:
        return None

    sq_a, sq_b = squares
    candidates = [
        chess.Move(sq_a, sq_b),
        chess.Move(sq_b, sq_a),
        # pawn promotion — default to queen
        chess.Move(sq_a, sq_b, promotion=chess.QUEEN),
        chess.Move(sq_b, sq_a, promotion=chess.QUEEN),
    ]
    for move in candidates:
        if move in board.legal_moves:
            return move

    return None


# ---------------------------------------------------------------------------
# CLOUD COMMUNICATION
# ---------------------------------------------------------------------------

def post_fen(fen: str) -> dict | None:
    """
    POST the current FEN to the Cloud Run endpoint.
    Expects a JSON response like:
        {"best_move": "e2e4", "explanation": "Controls the center..."}
    Returns the parsed JSON dict, or None on error.
    """
    try:
        response = requests.get(
            CLOUD_URL,
            params={"fen": fen, "time_limit": 1.0},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("[ERROR] Cloud request timed out.")
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] Cloud returned HTTP {e.response.status_code}")
    except Exception as e:
        print(f"[ERROR] Cloud request failed: {e}")
    return None


# ---------------------------------------------------------------------------
# GAME RESET
# ---------------------------------------------------------------------------

def reset_game():
    """Clear save files and return a fresh board and empty snapshot."""
    for f in (STATE_FILE, SNAPSHOT_FILE):
        if os.path.exists(f):
            os.remove(f)
    board = chess.Board()
    save_board(board)
    print("\n=== New game started — place pieces in starting position ===")
    print("Commands: Enter = capture move | \'new game\' = reset | \'quit\' = exit\n")
    return board, None


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main():
    board         = load_board()
    cap           = open_camera(CAMERA_INDEX)
    last_snapshot = None

    # Load persisted snapshot if one exists (e.g. after a crash)
    if os.path.exists(SNAPSHOT_FILE):
        last_snapshot = cv2.imread(SNAPSHOT_FILE)
        print(f"[INFO] Loaded previous snapshot from {SNAPSHOT_FILE}")

    print("\n=== Chess Vision — Jetson Nano A ===")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print("Commands: Enter = capture move | 'new game' = reset | 'quit' = exit\n")

    try:
        while True:
            if board.is_game_over():
                print("\n=== Game over ===")
                print(board.result())
                cmd = input("\nType 'new game' to play again or 'quit' to exit: ").strip().lower()
                if cmd == "new game":
                    board, last_snapshot = reset_game()
                    continue
                else:
                    break

            cmd = input(">> ").strip().lower()

            if cmd == "quit":
                cv2.destroyAllWindows()
                break

            if cmd == "new game":
                board, last_snapshot = reset_game()
                continue

            # anything else (including just pressing Enter) opens the preview
            show_preview(cap, board)

            # Take snapshot
            snapshot = get_warped_snapshot(cap)
            if snapshot is None:
                print("   Retake: board not detected. Try again.\n")
                continue

            # First move of the session — just establish baseline
            if last_snapshot is None:
                last_snapshot = snapshot
                cv2.imwrite(SNAPSHOT_FILE, last_snapshot)
                print("   Baseline snapshot saved. Make your first move.\n")
                continue

            # Find changed squares
            changed = find_changed_squares(last_snapshot, snapshot)
            print(f"   Changed squares (file,rank): {changed}")

            if len(changed) not in (2, 4):
                print(f"   Expected 2 changed squares (or 4 for castling), "
                      f"got {len(changed)}. Try again.\n")
                continue

            # Infer the move
            move = infer_move(board, changed)
            if move is None:
                print("   Could not match a legal move to those squares.")
                print("   Make sure the piece was moved cleanly and retry.\n")
                continue

            # Apply move
            san = board.san(move)       # e.g. "Nf3" — save before push
            board.push(move)
            last_snapshot = snapshot

            # Persist state
            save_board(board)
            cv2.imwrite(SNAPSHOT_FILE, last_snapshot)

            print(f"   Move detected: {san}  ({move.uci()})")
            print(f"   FEN: {board.fen()}\n")

            # Send to cloud, get best response move
            if not board.is_game_over():
                print("   Sending to cloud...")
                result = post_fen(board.fen())
                if result:
                    print(f"   Cloud best move : {result.get('best_move')}")
                    print(f"   Explanation     : {result.get('explanation')}\n")
                else:
                    print("   Cloud unavailable — continuing offline.\n")

    except KeyboardInterrupt:
        print("\n[INFO] Quit. Game state saved.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
