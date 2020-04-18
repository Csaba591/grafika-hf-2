//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Mészáros Csaba Máté
// Neptun : UY8Q7D
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char *const vertexSource =
        R"(
    #version 330				// Shader 3.3
    precision highp float;		// normal floats, makes no difference on desktop computers

    layout(location = 0) in vec2 cVertexPosition;
    out vec2 texcoord;

    void main() {
        texcoord = (cVertexPosition + vec2(1, 1)) / 2;
        gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);		// transform vp from modeling space to normalized device space
    }
)";

// fragment shader in GLSL
const char *const fragmentSource =
        R"(
    #version 330			// Shader 3.3
    precision highp float;	// normal floats, makes no difference on desktop computers

    uniform sampler2D textureUnit;
    in vec2 texcoord;
    out vec4 fragmentColor;

    void main() {
        fragmentColor = texture(textureUnit, texcoord);
    }
)";

GPUProgram gpuProgram; // vertex and fragment shaders

enum MaterialType { ROUGH, REFLECTIVE };
struct Material {
    vec3 ka, kd, ks;
    float shininess;
    vec3 F0;
    MaterialType type;
    Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
    RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
        ka = _kd * M_PI;
        kd = _kd; ks = _ks;
        shininess = _shininess;
    }
};

vec3 operator/(vec3 num, vec3 denom) {
    return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}
struct ReflectiveMaterial : Material {
    ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
        vec3 one(1, 1, 1);
        F0 = ((n - one)*(n - one) + kappa*kappa) / ((n + one)*(n + one) + kappa*kappa);
    }
};

struct Hit {
    float t;
    vec3 position, normal;
    Material * material;
    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) {
        start = _start;
        dir = normalize(_dir);
    }
};

class Quadratics {
    mat4 Q;
public:
    Quadratics(vec4 r1, vec4 r2, vec4 r3, vec4 r4) {
        Q = mat4(r1.x, r1.y, r1.z, r1.w,
                 r2.x, r2.y, r2.z, r2.w,
                 r3.x, r3.y, r3.z, r3.w,
                 r4.x, r4.y, r4.z, r4.w);
    }
    float f(vec4 x) {
        return dot(x * Q, x);
    }
    float fv(vec4 r, vec4 p) {
        return dot(r * Q, p);
    }
    vec3 gradf(vec4 x) {
        vec4 g = x * Q * 2;
        return vec3(g.x, g.y, g.z);
    }
};

class Intersectable {
protected:
    Material * material;
public:
    virtual Hit intersect(const Ray & ray) = 0;
};

class Sphere : public Intersectable, public Quadratics {
    vec3 orig;
    float r;
public:
    Sphere(vec3 _orig, float a, Material* _material)
    : Quadratics(
            vec4(1/(a*a)  , 0     , 0                          , 0                              ),
            vec4(0        , 1/(a*a),0                          , 0                              ),
            vec4(0        , 0     , 1/(a*a)                    , 0),
            vec4(0        , 0     , 0                          , -1)) {
        //printf("%f, %f, %f, %f\n", )
        orig = _orig;
        r = a;
        material = _material;
    }
    Hit intersect(const Ray & ray) override {
        /*Hit hit;
        vec4 S = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
        vec4 D = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
        float discr = (fv(D, S) + fv(S, D)) * (fv(D, S) + fv(S, D)) - (4 * f(D) * f(S));
        if(discr < 0) return hit;
        float t1 = (-(fv(D, S) + fv(S, D)) + sqrtf(discr)) / (2 * f(D));
        float t2 = (-(fv(D, S) + fv(S, D)) - sqrtf(discr)) / (2 * f(D));
        if(t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normalize(gradf(S + D * hit.t));
        hit.material = material;
        return hit;*/
        Hit hit;
        vec3 dist = ray.start - orig;
        float a = dot(ray.dir, ray.dir);
        float b = dot(dist, ray.dir) * 2.0f;
        float c = dot(dist, dist) - r * r;
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = (hit.position - orig) * (1.0f / r);
        hit.material = material;
        return hit;
    }
};

class Paraboloid : public Intersectable, public Quadratics {
public:
    Paraboloid(vec3 t, Material* _material)
            : Quadratics(
            vec4(-1,     0,    0,                             t.x),
            vec4( 0,     0,    0,                        -0.5f),
            vec4( 0,     0,   -1,                             t.z),
            vec4(  t.x, -0.5f,     t.z, -1 - t.x*t.x + t.y - t.z*t.z)) {
        material = _material;
    }
    Hit intersect(const Ray & ray) override {
        Hit hit;
        vec4 S = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
        vec4 D = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
        float discr = (fv(D, S) + fv(S, D)) * (fv(D, S) + fv(S, D)) - (4 * f(D) * f(S));
        if(discr < 0) return hit;
        float t1 = (-(fv(D, S) + fv(S, D)) + sqrtf(discr)) / (2 * f(D));
        float t2 = (-(fv(D, S) + fv(S, D)) - sqrtf(discr)) / (2 * f(D));
        if(t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normalize(gradf(S + D * hit.t));
        hit.material = material;
        return hit;
    }
};

class Cylinder : public Intersectable, public Quadratics {
    vec3 middle;
public:
    Cylinder(vec3 t, float a, Material* _material)
            : Quadratics(
            vec4(1/(a*a), 0,          0,            -1/a),
            vec4(      0, 0,          0,               0),
            vec4(      0, 0,    1/(a*a),      -t.z/(a*a)),
            vec4(   -1/a, 0, -t.z/(a*a), (t.z*t.z)/(a*a))) {
        middle = t;
        material = _material;
    }
    Hit intersect(const Ray & ray) override {
        Hit hit;
        vec4 S = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
        vec4 D = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
        float discr = (fv(D, S) + fv(S, D)) * (fv(D, S) + fv(S, D)) - (4 * f(D) * f(S));
        if(discr < 0) return hit;
        float t1 = (-(fv(D, S) + fv(S, D)) + sqrtf(discr)) / (2 * f(D));
        float t2 = (-(fv(D, S) + fv(S, D)) - sqrtf(discr)) / (2 * f(D));
        if(t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normalize(gradf(S + D * hit.t));
        hit.material = material;
        return hit;
    }
};

class Ellipsoid : public Intersectable, public Quadratics {
    vec3 orig;
public:
    Ellipsoid(vec3 t, vec3 a, Material* _material)
            : Quadratics(
            vec4(   1/(a.x*a.x),             0,             0, -t.x/(a.x*a.x)),
            vec4(             0,   1/(a.y*a.y),             0, -t.y/(a.y*a.y)),
            vec4(             0,             0,   1/(a.z*a.z), -t.z/(a.z*a.z)),
            vec4(-t.x/(a.x*a.x),-t.y/(a.y*a.y),-t.z/(a.z*a.z), -1 + (t.x*t.x)/(a.x*a.x) + (t.y*t.y)/(a.y*a.y) + (t.z*t.z)/(a.z*a.z))) {
        orig = t;
        material = _material;
    }
    Hit intersect(const Ray & ray) override {
        Hit hit;
        vec4 S = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
        vec4 D = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
        float discr = (fv(D, S) + fv(S, D)) * (fv(D, S) + fv(S, D)) - (4 * f(D) * f(S));
        if(discr < 0) return hit;
        float t1 = (-(fv(D, S) + fv(S, D)) + sqrtf(discr)) / (2 * f(D));
        float t2 = (-(fv(D, S) + fv(S, D)) - sqrtf(discr)) / (2 * f(D));
        if(t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        vec4 plane(0, 1, 0, -orig.y-0.993F);
        vec4 currPos4(hit.position.x, hit.position.y, hit.position.z, 1);
        if(sqrtf(fabs(hit.position.x)*fabs(hit.position.x) + fabs(hit.position.z)*fabs(hit.position.z)) <= 0.2F && hit.position.y > 0) {//dot(plane, currPos4) > 0) {
            hit.t = -1;
            return hit;
        }
        hit.normal = normalize(gradf(S + D * hit.t));
        hit.material = material;
        return hit;
    }
};

class Hyperboloid : public Intersectable, public Quadratics {
    vec3 orig;
    float height;
public:
    Hyperboloid(vec3 t, float h, float a, float _height, Material* _material)
            : Quadratics(
            vec4(   1/(a*a),         0,          0,      -t.x/(a*a)),
            vec4(         0,  -1/(h*h),          0,       t.y/(h*h)),
            vec4(         0,         0,    1/(a*a),      -t.z/(a*a)),
            vec4(-t.x/(a*a), t.y/(h*h), -t.z/(a*a), -1 - (t.y*t.y)/(h*h) + (t.x*t.x)/(a*a) + (t.z*t.z)/(a*a))) {
        material = _material;
        orig = t;
        height = _height;
    }
    Hit intersect(const Ray & ray) override {
        Hit hit;
        vec4 S = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
        vec4 D = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
        float discr = (fv(D, S) + fv(S, D)) * (fv(D, S) + fv(S, D)) - (4 * f(D) * f(S));
        if(discr < 0) return hit;
        float t1 = (-(fv(D, S) + fv(S, D)) + sqrtf(discr)) / (2 * f(D));
        float t2 = (-(fv(D, S) + fv(S, D)) - sqrtf(discr)) / (2 * f(D));
        if(t1 <= 0 && t2 <= 0) return hit;
        hit.t = (t2 < t1) ? t2 : t1;
        float tt = (t2 < t1) ? t1 : t2;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normalize(gradf(S + D * hit.t));
        vec4 plane1(0, 1, 0, -orig.y-height);
        vec4 plane2(0, 1, 0, -orig.y);
        vec4 currPos4(hit.position.x, hit.position.y, hit.position.z, 1);
        if((dot(plane1, currPos4) > 0 || dot(plane2, currPos4) < 0)) {
            hit.t = tt;
            hit.position = ray.start + ray.dir * hit.t;
            hit.normal = normalize(gradf(S + D * hit.t));
            currPos4 = vec4(hit.position.x, hit.position.y, hit.position.z, 1);
        }
        if((dot(plane1, currPos4) > 0 || dot(plane2, currPos4) < 0)) {
            hit.t = -1;
            return hit;
        }
        hit.material = material;
        return hit;
    }
    vec3 Orig() { return orig; }
};

class Plane : public Intersectable {
    vec3 orig, normal;
public:
    Plane(vec3 _orig, Material* _material, vec3 _normal = vec3(0, 1, 0)) {
        orig = _orig;
        material = _material;
        normal = _normal;
    }
    Hit intersect(const Ray & ray) override {
        // forras: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
        Hit hit;
        float div = dot(ray.dir, normal);
        if (fabs(div) > 0.0001f) {
            float t = dot((orig - ray.start), normal) / div;
            if (t > 0) hit.t = t;
        }
        else return hit;
        if(hit.t <= 0) return hit;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normal;
        hit.material = material;
        return hit;
    }
};

struct Light {
    vec3 dir, Le;
    Light(vec3 _dir, vec3 _Le) {
        dir = normalize(_dir * (-1)); Le = _Le;
    }
};

class Camera {
    vec3 eye, lookAt, up, right;
    float fov;
public:
    Camera(vec3 _eye = vec3(0.F, 0.F, -1.5F),
           vec3 _lookAt = vec3(0.F, 0.F, -0.5F), float _fov = 90.F) {
        eye = _eye; lookAt = _lookAt; fov = _fov * static_cast<float>(M_PI) / 180;
        vec3 vup(0, 1, 0);
        vec3 w = lookAt - eye;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }
    Ray getRay(const int X, const int Y) {
        vec3 dir = lookAt - eye
                   + right * (2 * static_cast<float>(X) / static_cast<float>(windowWidth) - 1.F)
                   + up * (2 * static_cast<float>(Y) / static_cast<float>(windowHeight) - 1.F);
//        vec3 dir = lookAt + right * (2 * (X + 0.5F) / windowWidth - 1) + up * (2 * (Y + 0.5F) / windowHeight - 1) - eye;
        return Ray(eye, dir);
    }
};

class Canvas {
    unsigned int vao, textureId;
public:
    Canvas() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        unsigned int vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float vertexCoords[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    void loadTexture(std::vector<vec4> & image) {
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
    }
    void draw() {
        glBindVertexArray(vao);
        int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
        const unsigned int textureUnit = 0;
        if(location >= 0) {
            glUniform1i(location, textureUnit);
            glActiveTexture(GL_TEXTURE0 + textureUnit);
            glBindTexture(GL_TEXTURE_2D, textureId);
        }
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

float holeR = 0.2F;
float holeY = 0.95F;

float rnd() { return ((static_cast<float>(rand()) / RAND_MAX) * 2 - 1) * holeR; }

vec2 rndXZ() {
    vec2 res(rnd(), rnd());
    while(length(res) > holeR) {
        res.x = rnd();
        res.y = rnd();
    }
    return res;
}

class Scene {
    std::vector<Intersectable *> objects;
    std::vector<Light *> lights;
    vec3 sunPos, sun, sky;
    Camera *camera;
    vec3 La;
public:
    void build() {
        camera = new Camera();

        vec3 lightDirection(-1, -1, 1), Le(2, 2, 2);
        lights.push_back(new Light(lightDirection, Le));

        sunPos = vec3(1, 2, 1);
        sun = vec3(0.4F, 0.38F, 0.17F);
        sky = vec3(0.16F, 0.25F, 0.4F);
        La = vec3(0.4f, 0.4f, 0.4f);

        vec3 ks(2, 2, 2);
        auto * green = new RoughMaterial(vec3(0.1f, 0.4f, 0.1f), ks, 300);
        auto * purple = new RoughMaterial(vec3(0.4f, 0.2f, 0.3f), ks, 50);
        // gold
        vec3 n(0.17f, 0.35f, 1.5f), kappa(3.1f, 2.7f, 1.9f);
        ReflectiveMaterial * gold = new ReflectiveMaterial(n, kappa);
        //silver
        vec3 n_s(0.14f, 0.16f, 0.13f), kappa_s(4.1f, 2.3f, 3.1f);
        ReflectiveMaterial * silver = new ReflectiveMaterial(n_s, kappa_s);
        objects.push_back(new Plane(vec3(0, -0.5f, 0), purple));
        //objects.push_back(new Plane(vec3(0, 0, 6), purple, vec3(0, 0, -1)));
        //objects.push_back(new Plane(vec3(0, 1, 0), purple, vec3(0, -1, 0)));
        objects.push_back(new Ellipsoid(vec3(0, 0, 0), vec3(1.4f, 1, 2), green));
        objects.push_back(new Sphere(vec3(0, 0, 0.5f), 0.2f, gold));
        objects.push_back(new Sphere(vec3(0.4f, 0.2f, 0.5f), 0.3f, purple));
        //objects.push_back(new Cylinder(vec3(-4.f, 0, 0), 0.2f, green));
        objects.push_back(new Hyperboloid(vec3(0, holeY, 0), 0.6f, holeR, 1.05F, silver));
        objects.push_back(new Hyperboloid(vec3(-0.8f, 0.2f, 0), 0.6f, holeR, 1.05F, purple));
    }
    void render(std::vector<vec4> & image) {
        for(int Y = 0; Y < windowHeight; Y++) {
            for(int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera->getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }
    vec3 trace(Ray ray, int depth = 0) {
        if(depth > 5) return La;
        Hit hit = firstIntersect(ray);
        if(hit.t < 0) return La;

        const float epsilon = 0.0001F;
        vec3 color(0, 0, 0);
        if(hit.material->type == ROUGH) {
            vec3 sum(0, 0, 0);
            Hit h = hit;
            Ray newRay = ray;
            vec2 coord(0, 0);

            vec3 holeDir = vec3(coord.x, holeY, coord.y) - hit.position;
            Ray shadowRay(hit.position + hit.normal * epsilon, holeDir);
            float cosTheta = dot(hit.normal, holeDir);
            if (cosTheta > 0 && !shadowIntersect(shadowRay)) {    // shadow computation
                for (int i = 0; i < 4; i++) {
                    coord = rndXZ();
                    newRay.start = hit.position + hit.normal * epsilon;
                    newRay.dir = normalize(vec3(coord.x, holeY, coord.y) - newRay.start);
                    for (int j = 0; j < 10; j++) {
                        h = firstIntersect(newRay);
                        if (h.t < 0) break;
                        newRay.start = h.position + h.normal * epsilon;
                        newRay.dir = normalize(newRay.dir - h.normal * dot(h.normal, newRay.dir) * 2.0f);
                    }
                    vec3 sunDir = normalize(sunPos - newRay.start);
                    sum = sum + sky + sun * powf(dot(newRay.dir, sunDir), 10);
                    //printf("%f, %f, %f\n", sum.x, sum.y, sum.z);
                }
                color = color + sum;
            } else {
                color = hit.material->ka * La;
            }
            /*for (auto &light : lights) {
                Ray shadowRay(hit.position + hit.normal * epsilon, light->dir);
                float cosTheta = dot(hit.normal, light->dir);
                if (cosTheta > 0 && !shadowIntersect(shadowRay)) {    // shadow computation
                    color = color + light->Le * hit.material->kd * cosTheta;
                    vec3 halfway = normalize(-ray.dir + light->dir);
                    float cosDelta = dot(hit.normal, halfway);
                    if (cosDelta > 0)
                        color = color + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
                }
            }*/
        } else if(hit.material->type == REFLECTIVE) {
            vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.F;
            float cosa = -dot(ray.dir, hit.normal);
            vec3 one(1, 1, 1);
            vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);
            color = color + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth+1) * F;
        }
        //printf("%f, %f, %f\n", color.x, color.y, color.z);
        return color;
    }
    bool shadowIntersect(Ray ray) {	// for directional lights
        for (auto & object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }
    Hit firstIntersect(const Ray & ray) {
        Hit hit;
        Hit test;
        for(auto & o : objects) {
            test = o->intersect(ray);
            if(test.t > 0 && (hit.t < 0 || test.t < hit.t))
                hit = test;
        }
        if(dot(ray.dir, hit.normal) > 0) hit.normal = hit.normal * (-1);
        return hit;
    }
};

Scene scene;
Canvas *canvas;
// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);

    scene.build();
    canvas = new Canvas();

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0); // background color
    glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

    std::vector<vec4> image(windowWidth * windowHeight);
    scene.render(image);
    canvas->loadTexture(image);
    canvas->draw();

    glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(const unsigned char key, int /*pX*/, int /*pY*/) {
    if (key == 'd') {
        glutPostRedisplay();
    }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char /*key*/, int /*pX*/, int /*pY*/) {
}

// Move mouse with key pressed
void onMouseMotion(const int pX, const int pY) {
}

// Mouse click event
void onMouse(const int button, const int state, const int pX, const int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
