use std::cell::{Cell, RefCell};
use std::fmt;

pub struct Matrix2D<T>
where
    T: std::marker::Copy,
{
    rows: Cell<u32>,
    cols: Cell<u32>,
    data: RefCell<Vec<Vec<T>>>,
}

impl<T> Matrix2D<T>
where
    T: std::marker::Copy,
{
    pub fn new(rows: u32, cols: u32, default: T) -> Self {
        Self {
            rows: Cell::new(rows),
            cols: Cell::new(cols),
            data: RefCell::new(std::vec::from_elem(
                std::vec::from_elem(default, cols as usize),
                rows as usize,
            )),
        }
    }

    pub fn rows(&self) -> u32 {
        self.rows.get()
    }

    pub fn cols(&self) -> u32 {
        self.cols.get()
    }

    pub fn get(&self, row: i32, col: i32) -> T {
        self.data.borrow()[row as usize][col as usize]
    }

    pub fn set(&mut self, row: i32, col: i32, val: T) {
        self.data.get_mut()[row as usize][col as usize] = val;
    }
}

impl<T> fmt::Display for Matrix2D<T>
where
    T: std::marker::Copy + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut str: String = "".to_owned();
        str.push_str(&format!(
            "Matrix2D, Rows: {}, Columns: {}\n",
            self.rows(),
            self.cols()
        ));
        for row in 0..self.rows() {
            str.push_str("[");
            for col in 0..self.rows() {
                str.push_str(&format!(
                    "{}",
                    self.data.borrow()[row as usize][col as usize]
                ));

                if col != self.rows() - 1 {
                    str.push_str("\t");
                }
            }
            str.push_str("]\n");
        }
        write!(f, "{}", str)
    }
}
