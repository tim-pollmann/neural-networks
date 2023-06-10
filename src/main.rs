mod matrix2d;

fn main() {
    let mut m = matrix2d::Matrix2D::new(4, 5, 77.8);
    m.set(2, 1, 4.9);
    
    print!("{}", m.get(2, 1));
    print!("{}", m);
}
